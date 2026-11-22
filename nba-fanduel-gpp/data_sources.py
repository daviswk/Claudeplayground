"""Data ingestion: ESPN box scores, FanDuel slate CSV, Databallr exports, NBA injury PDF, The Odds API.

Network fetchers (Databallr live scrape, NBA injury PDF download, The Odds API) import their
HTTP/PDF libraries lazily so the offline pieces work with numpy/pandas/scipy alone.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import FD_SCORING, INJURY_OUT_PROB, PROP_SD_PER_SQRT
from normalize import (
    TEAM_FULL_NAMES,
    audit_join_rate,
    normalize_game,
    normalize_player_name,
    normalize_team,
    parse_fd_positions,
)

logger = logging.getLogger(__name__)

STATS = ["pts", "reb", "ast", "stl", "blk", "tov"]
TEAM_FACTOR_COLS = [
    "team", "ortg", "drtg", "net", "pace", "ots", "otov", "orb", "dts", "dtov", "drb",
    "three_p_luck", "stop", "def_rim_freq", "def_ftr",
]


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def compute_fd_points(df: pd.DataFrame) -> pd.Series:
    """Exact FanDuel NBA points from box-score columns (missing columns count as 0)."""
    total = pd.Series(0.0, index=df.index)
    for stat, w in FD_SCORING.items():
        if stat in df.columns:
            total = total + w * pd.to_numeric(df[stat], errors="coerce").fillna(0.0)
    return total


# --------------------------------------------------------------------------------------
# ESPN (same JSON layout nba-props/fetch_espn.py writes: events.json + sum_<eid>.json)
# --------------------------------------------------------------------------------------
def _made_att(s: str) -> Tuple[int, int]:
    try:
        m, a = str(s).split("-")
        return int(m), int(a)
    except ValueError:
        return 0, 0


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def build_espn_fanduel_logs(espn_dir) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse ESPN summaries into (player_logs, games).

    player_logs: one row per player-game with FanDuel points, shooting/rebounding splits,
      team-perspective spread (negative = favourite), rest/back-to-back/3-in-4/altitude flags.
    games: one row per event (final and scheduled) with total and home spread.
    """
    espn_dir = Path(espn_dir)
    events = json.loads((espn_dir / "events.json").read_text())
    rows, games = [], []
    for eid, meta in events.items():
        path = espn_dir / f"sum_{eid}.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        comp = s["header"]["competitions"][0]
        sides = {c["homeAway"]: normalize_team(c["team"]["abbreviation"]) for c in comp["competitors"]}
        home, away = sides.get("home", ""), sides.get("away", "")
        pc = (s.get("pickcenter") or [{}])[0]
        total, home_spread = pc.get("overUnder"), pc.get("spread")
        date = dt.datetime.strptime(str(meta["date"]), "%Y%m%d").strftime("%Y-%m-%d")
        games.append({"eid": str(eid), "date": date, "home_team": home, "away_team": away, "total": total,
                      "home_spread": home_spread, "status": meta.get("status", "")})
        if meta.get("status") != "STATUS_FINAL":
            continue
        for tb in s.get("boxscore", {}).get("players", []):
            team = normalize_team(tb["team"]["abbreviation"])
            if not tb.get("statistics"):
                continue
            st = tb["statistics"][0]
            keys = st["keys"]
            for ath in st.get("athletes", []):
                if ath.get("didNotPlay") or not ath.get("stats"):
                    continue
                v = dict(zip(keys, ath["stats"]))
                mins = _num(v.get("minutes"))
                if mins <= 0:
                    continue
                fgm, fga = _made_att(v.get("fieldGoalsMade-fieldGoalsAttempted", "0-0"))
                tpm, tpa = _made_att(v.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", "0-0"))
                ftm, fta = _made_att(v.get("freeThrowsMade-freeThrowsAttempted", "0-0"))
                is_home = int(team == home)
                rows.append({
                    "eid": str(eid), "date": date, "player_id": ath["athlete"].get("id"),
                    "name": ath["athlete"]["displayName"],
                    "player_key": normalize_player_name(ath["athlete"]["displayName"]),
                    "pos": (ath["athlete"].get("position") or {}).get("abbreviation", ""),
                    "team": team, "opp": away if is_home else home, "home": is_home,
                    "starter": int(bool(ath.get("starter"))), "min": mins,
                    "pts": _num(v.get("points")), "reb": _num(v.get("rebounds")), "ast": _num(v.get("assists")),
                    "stl": _num(v.get("steals")), "blk": _num(v.get("blocks")), "tov": _num(v.get("turnovers")),
                    "pf": _num(v.get("fouls")), "plus_minus": _num(v.get("plusMinus")),
                    "oreb": _num(v.get("offensiveRebounds")), "dreb": _num(v.get("defensiveRebounds")),
                    "fgm": fgm, "fga": fga, "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta,
                    "total": total,
                    "home_spread": home_spread,
                    "spread": (home_spread if is_home else -home_spread) if home_spread is not None else np.nan,
                })
    logs = pd.DataFrame(rows)
    games_df = pd.DataFrame(games)
    if logs.empty:
        return logs, games_df
    logs["fd_pts"] = compute_fd_points(logs)
    logs = add_schedule_flags(logs)
    return logs.sort_values(["date", "eid"]).reset_index(drop=True), games_df


def add_schedule_flags(logs: pd.DataFrame) -> pd.DataFrame:
    """rest_days, is_b2b, is_3in4 per team-game; at_altitude when the game is in Denver or Utah."""
    tg = logs[["team", "eid", "date", "home", "opp"]].drop_duplicates(["team", "eid"]).copy()
    tg["d"] = pd.to_datetime(tg["date"])
    tg = tg.sort_values(["team", "d"])
    tg["rest_days"] = tg.groupby("team")["d"].diff().dt.days.fillna(3).clip(upper=5)
    tg["is_b2b"] = (tg["rest_days"] == 1).astype(int)
    tg["is_3in4"] = (tg.groupby("team")["d"].diff(2).dt.days <= 3).astype(int)
    game_home = np.where(tg["home"] == 1, tg["team"], tg["opp"])
    tg["at_altitude"] = pd.Series(game_home, index=tg.index).isin(["DEN", "UTA"]).astype(int)
    return logs.merge(tg[["team", "eid", "rest_days", "is_b2b", "is_3in4", "at_altitude"]], on=["team", "eid"], how="left")


def derive_team_factors_from_logs(logs: pd.DataFrame) -> pd.DataFrame:
    """Databallr-style six-factor table computed from ESPN box scores.

    Units are percentage points relative to league average. Signs: positive is good for that
    team (ots/otov/orb on offense; dts/dtov/drb/stop on defense). three_p_luck > 0 means opponents
    have shot below the league 3P% against this team (expect regression upward).
    def_rim_freq is not derivable from box scores and is left NaN.
    """
    need = {"fga", "fta", "oreb", "dreb", "tov", "tpm", "tpa", "stl", "blk", "pts", "min"}
    if logs is None or logs.empty or not need.issubset(logs.columns):
        return pd.DataFrame(columns=TEAM_FACTOR_COLS)
    t = logs.groupby(["eid", "team", "opp"])[sorted(need)].sum().reset_index()
    o = t.drop(columns=["opp"]).rename(columns={c: f"o_{c}" for c in need}).rename(columns={"team": "opp"})
    t = t.merge(o, on=["eid", "opp"], how="inner")
    t["poss"] = t.fga + 0.44 * t.fta - t.oreb + t.tov
    t["o_poss"] = t.o_fga + 0.44 * t.o_fta - t.o_oreb + t.o_tov
    agg = t.groupby("team").sum(numeric_only=True)

    def ts(p, fga, fta):
        return p / (2 * (fga + 0.44 * fta))

    lg = agg.sum()
    lg_ts = ts(lg.pts, lg.fga, lg.fta)
    lg_tov = lg.tov / lg.poss
    lg_orb = lg.oreb / (lg.oreb + lg.o_dreb)
    lg_3p = lg.tpm / max(lg.tpa, 1)
    lg_stop = (lg.o_tov + lg.blk) / lg.o_poss
    out = pd.DataFrame(index=agg.index)
    out["ortg"] = 100 * agg.pts / agg.poss
    out["drtg"] = 100 * agg.o_pts / agg.o_poss
    out["net"] = out.ortg - out.drtg
    out["pace"] = 48 * ((agg.poss + agg.o_poss) / 2) / (agg["min"] / 5)
    out["ots"] = 100 * (ts(agg.pts, agg.fga, agg.fta) - lg_ts)
    out["otov"] = -100 * (agg.tov / agg.poss - lg_tov)
    out["orb"] = 100 * (agg.oreb / (agg.oreb + agg.o_dreb) - lg_orb)
    out["dts"] = -100 * (ts(agg.o_pts, agg.o_fga, agg.o_fta) - lg_ts)
    out["dtov"] = 100 * (agg.o_tov / agg.o_poss - lg_tov)
    out["drb"] = 100 * (agg.dreb / (agg.dreb + agg.o_oreb) - (1 - lg_orb))
    out["three_p_luck"] = -100 * (agg.o_tpm / agg.o_tpa.clip(lower=1) - lg_3p)
    out["stop"] = 100 * ((agg.o_tov + agg.blk) / agg.o_poss - lg_stop)
    out["def_rim_freq"] = np.nan
    out["def_ftr"] = agg.o_fta / agg.o_fga
    return out.reset_index().rename(columns={"index": "team"})[TEAM_FACTOR_COLS]


# --------------------------------------------------------------------------------------
# FanDuel slate
# --------------------------------------------------------------------------------------
def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lookup = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for n in names:
        key = re.sub(r"[^a-z0-9]", "", n.lower())
        if key in lookup:
            return lookup[key]
    return None


def load_fanduel_slate(csv_path) -> pd.DataFrame:
    """Load FanDuel's 'Download players list' CSV for an NBA Classic contest."""
    raw = pd.read_csv(csv_path)
    c_id = _col(raw, "Id", "Player ID")
    c_pos = _col(raw, "Position", "Roster Position")
    c_nick = _col(raw, "Nickname", "Name")
    c_first, c_last = _col(raw, "First Name"), _col(raw, "Last Name")
    c_sal, c_fppg = _col(raw, "Salary"), _col(raw, "FPPG")
    c_game, c_team, c_opp = _col(raw, "Game"), _col(raw, "Team"), _col(raw, "Opponent")
    c_inj = _col(raw, "Injury Indicator")
    if not (c_id and c_pos and c_sal and c_team):
        raise ValueError(f"{csv_path}: not a FanDuel NBA players CSV (columns: {list(raw.columns)})")
    if c_nick:
        names = raw[c_nick].astype(str)
    else:
        names = raw[c_first].astype(str).str.strip() + " " + raw[c_last].astype(str).str.strip()
    team = raw[c_team].map(normalize_team)
    opp = raw[c_opp].map(normalize_team) if c_opp else pd.Series("", index=raw.index)
    game = raw[c_game].map(normalize_game) if c_game else pd.Series("", index=raw.index)
    df = pd.DataFrame({
        "fd_id": raw[c_id].astype(str),
        "name": names,
        "player_key": names.map(normalize_player_name),
        "pos_raw": raw[c_pos].astype(str),
        "salary": pd.to_numeric(raw[c_sal], errors="coerce").fillna(0).astype(int),
        "fppg": pd.to_numeric(raw[c_fppg], errors="coerce").fillna(0.0) if c_fppg else 0.0,
        "team": team,
        "opp": opp,
        "game": game,
        "fd_injury": raw[c_inj].fillna("").astype(str).str.strip().str.upper() if c_inj else "",
    })
    df["positions"] = df["pos_raw"].map(parse_fd_positions)
    df["primary_pos"] = df["positions"].map(lambda p: p[0] if p else "")
    missing_game = df["game"].eq("") & df["opp"].ne("")
    df.loc[missing_game, "game"] = df.loc[missing_game, "team"] + "@" + df.loc[missing_game, "opp"]
    return df[df["salary"] > 0].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Databallr (CSV exports; live scrape is best-effort)
# --------------------------------------------------------------------------------------
_DB_TEAM_COLS = {
    "team": "team", "tm": "team", "teamname": "team", "ortg": "ortg", "drtg": "drtg", "net": "net", "netrtg": "net",
    "ots": "ots", "otov": "otov", "orb": "orb", "dts": "dts", "dtov": "dtov", "drb": "drb",
    "3pluck": "three_p_luck", "threepluck": "three_p_luck", "3pl": "three_p_luck", "stop": "stop", "pace": "pace",
    "defrimfreq": "def_rim_freq", "rimfreq": "def_rim_freq", "defftr": "def_ftr", "dftr": "def_ftr",
}


def _normalize_db_teams(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        key = re.sub(r"[^a-z0-9]", "", str(c).lower())
        if key in _DB_TEAM_COLS and _DB_TEAM_COLS[key] not in ren.values():
            ren[c] = _DB_TEAM_COLS[key]
    out = df.rename(columns=ren)
    if "team" not in out.columns:
        raise ValueError(f"Databallr team table has no team column: {list(df.columns)}")
    out["team"] = out["team"].map(normalize_team)
    for c in TEAM_FACTOR_COLS[1:]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c].astype(str).str.replace("%", "").str.replace("+", ""), errors="coerce")
    return out[TEAM_FACTOR_COLS]


def _scrape_databallr_teams(url: str = "https://databallr.com/teams") -> pd.DataFrame:
    """Best-effort: an HTML <table>, else a JSON array of team rows embedded in the page."""
    import requests
    from bs4 import BeautifulSoup

    html = requests.get(url, timeout=25, headers={"User-Agent": "nba-fanduel-gpp/1.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is not None:
        header = [th.get_text(strip=True) for th in table.find_all("th")]
        body = [[td.get_text(strip=True) for td in tr.find_all("td")] for tr in table.find_all("tr")]
        body = [r for r in body if len(r) == len(header)]
        if body:
            return _normalize_db_teams(pd.DataFrame(body, columns=header))
    for script in soup.find_all("script"):
        txt = script.string or ""
        if "drtg" not in txt.lower():
            continue
        for m in re.finditer(r"\[\{.*?\}\]", txt, flags=re.S):
            try:
                rows = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list) and len(rows) >= 20 and isinstance(rows[0], dict):
                try:
                    return _normalize_db_teams(pd.DataFrame(rows))
                except ValueError:
                    continue
    raise RuntimeError("Could not find a team table on databallr.com/teams; export the table to CSV and pass --databallr-teams-csv.")


def fetch_databallr_teams(csv_path: str = "", live: bool = False) -> pd.DataFrame:
    """Databallr team six-factor table from a CSV export, or a live scrape when `live`."""
    if csv_path:
        return _normalize_db_teams(pd.read_csv(csv_path))
    if live:
        try:
            return _scrape_databallr_teams()
        except Exception as exc:  # network/layout failure falls back to ESPN-derived factors upstream
            logger.warning("Databallr live fetch failed (%s); falling back to ESPN-derived team factors.", exc)
    return pd.DataFrame(columns=TEAM_FACTOR_COLS)


def load_databallr_players(csv_path: str = "") -> pd.DataFrame:
    """Databallr player stats / DPM export -> player_key, team, dpm, o_dpm, d_dpm, zts, rapm, usg_db."""
    cols = ["player_key", "team", "dpm", "o_dpm", "d_dpm", "zts", "rapm", "usg_db"]
    if not csv_path:
        return pd.DataFrame(columns=cols)
    raw = pd.read_csv(csv_path)
    name_c = _col(raw, "Player", "Name", "player_name")
    if not name_c:
        raise ValueError(f"{csv_path}: no player/name column")
    out = pd.DataFrame({"player_key": raw[name_c].map(normalize_player_name)})
    team_c = _col(raw, "Team", "Tm")
    out["team"] = raw[team_c].map(normalize_team) if team_c else ""
    for dst, cands in {"dpm": ("DPM",), "o_dpm": ("O-DPM", "ODPM"), "d_dpm": ("D-DPM", "DDPM"), "zts": ("zTS",),
                       "rapm": ("RAPM",), "usg_db": ("USG", "USG%", "Usage")}.items():
        c = _col(raw, *cands)
        out[dst] = pd.to_numeric(raw[c], errors="coerce") if c else np.nan
    return out.drop_duplicates("player_key")


def load_databallr_wowy(csv_path="", prior_minutes: float = 266.0) -> pd.DataFrame:
    """WOWY / Stat Line Shift export: FanDuel per-36 shift when `teammate_out` sits, shrunk by
    off_min / (off_min + padding). Accepts a ready `fd_p36_shift` column or per-stat per-36 shifts."""
    cols = ["player_key", "teammate_out_key", "team", "off_min", "shrink_weight", "fd_p36_shift_shrunk"]
    if not csv_path:
        return pd.DataFrame(columns=cols)
    raw = pd.read_csv(csv_path)
    p_c = _col(raw, "player", "Player", "name")
    o_c = _col(raw, "teammate_out", "teammate", "without", "off_player", "teammate off")
    m_c = _col(raw, "off_min", "min_off", "minutes_without", "off minutes")
    if not (p_c and o_c and m_c):
        raise ValueError(f"{csv_path}: need player, teammate_out and off_min columns")
    out = pd.DataFrame({
        "player_key": raw[p_c].map(normalize_player_name),
        "teammate_out_key": raw[o_c].map(normalize_player_name),
        "off_min": pd.to_numeric(raw[m_c], errors="coerce").fillna(0.0),
    })
    t_c = _col(raw, "team", "Tm")
    out["team"] = raw[t_c].map(normalize_team) if t_c else ""
    fd_c = _col(raw, "fd_p36_shift", "fd_p36_diff")
    if fd_c:
        shift = pd.to_numeric(raw[fd_c], errors="coerce").fillna(0.0)
    else:
        shift = pd.Series(0.0, index=raw.index)
        for stat, w in FD_SCORING.items():
            c = _col(raw, f"{stat}_p36_diff", f"{stat}_p36_shift", f"{stat} shift", f"{stat}_shift", f"{stat}_diff")
            if c:
                shift = shift + w * pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
    out["shrink_weight"] = out["off_min"] / (out["off_min"] + prior_minutes)
    out["fd_p36_shift_shrunk"] = shift * out["shrink_weight"]
    return out[cols]


# --------------------------------------------------------------------------------------
# NBA official injury report
# --------------------------------------------------------------------------------------
_W = r"[A-Z][A-Za-z'\.\-]+"
_STATUS_RE = re.compile(
    rf"(?P<name>{_W}(?: {_W})?(?: (?:Jr\.?|Sr\.?|II|III|IV))?, {_W}(?: {_W})?)\s+"
    r"(?P<status>Out|Doubtful|Questionable|Probable|Available)\b"
)
_TEAM_RE = re.compile("|".join(sorted((re.escape(k) for k in TEAM_FULL_NAMES), key=len, reverse=True)))


def parse_injury_report_text(text: str) -> pd.DataFrame:
    """Parse the text layer of the official NBA injury report into player/team/status rows."""
    rows = []
    team = ""
    for line in text.splitlines():
        tm = _TEAM_RE.search(line)
        if tm:
            team = TEAM_FULL_NAMES[tm.group(0)]
            line = _TEAM_RE.sub(" | ", line)
        for m in _STATUS_RE.finditer(line):
            reason = line[m.end():].strip()
            rows.append({
                "player_name": m.group("name").strip(),
                "player_key": normalize_player_name(m.group("name")),
                "team": team,
                "status": m.group("status").upper(),
                "reason": reason,
            })
    df = pd.DataFrame(rows, columns=["player_name", "player_key", "team", "status", "reason"])
    df["out_prob"] = df["status"].map(INJURY_OUT_PROB).fillna(0.0)
    return df.drop_duplicates("player_key", keep="last").reset_index(drop=True)


def parse_nba_injury_pdf(pdf_path) -> pd.DataFrame:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    return parse_injury_report_text(text)


def fetch_nba_injury_report(slate_date: str, out_dir, lookback_hours: int = 30) -> pd.DataFrame:
    """Download the latest official injury report PDF published for `slate_date` and parse it.

    The league posts reports at ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>.pdf.
    Both the hourly ('05PM') and quarter-hour ('05_30PM') file-name styles are tried, newest first.
    """
    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{t}.pdf"
    day = dt.datetime.strptime(slate_date, "%Y-%m-%d")
    end = min(dt.datetime.utcnow() - dt.timedelta(hours=5), day + dt.timedelta(hours=23, minutes=45))
    cursor = end.replace(minute=(end.minute // 15) * 15, second=0, microsecond=0)
    tried = 0
    while tried < lookback_hours * 4 and cursor >= day - dt.timedelta(hours=6):
        stamp_d = cursor.strftime("%Y-%m-%d")
        h12 = cursor.strftime("%I%p")
        names = [f"{h12[:2]}_{cursor.minute:02d}{h12[2:]}"]
        if cursor.minute == 0:
            names.append(h12)
        for t in names:
            url = base.format(d=stamp_d, t=t)
            try:
                r = requests.get(url, timeout=20)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                path = out_dir / Path(url).name
                path.write_bytes(r.content)
                logger.info("Injury report: %s", url)
                return parse_nba_injury_pdf(path)
        cursor -= dt.timedelta(minutes=15)
        tried += 1
    logger.warning("No official injury report found for %s; using FanDuel injury indicators only.", slate_date)
    return pd.DataFrame(columns=["player_name", "player_key", "team", "status", "reason", "out_prob"])


# --------------------------------------------------------------------------------------
# The Odds API (game lines + player props)
# --------------------------------------------------------------------------------------
PROP_MARKETS = {
    "player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
    "player_steals": "stl", "player_blocks": "blk", "player_turnovers": "tov",
}


def fetch_odds_api_nba(max_credits: int = 120, min_remaining: int = 50, api_key: Optional[str] = None,
                       regions: str = "us") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Game totals/spreads and player props from The Odds API v4, with a credit circuit breaker.

    Each request costs (#markets x #regions) credits; the run stops before exceeding `max_credits`
    or dropping the account below `min_remaining`. Needs ODDS_API_KEY.
    """
    import requests

    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("Set ODDS_API_KEY to use --live-fetch odds.")
    base = "https://api.the-odds-api.com/v4/sports/basketball_nba"
    n_regions = len(regions.split(","))
    state = {"spent": 0, "remaining": None}

    def get(path: str, markets: List[str]):
        cost = len(markets) * n_regions
        if state["spent"] + cost > max_credits:
            raise _CreditStop(f"would exceed max_credits={max_credits}")
        if state["remaining"] is not None and state["remaining"] - cost < min_remaining:
            raise _CreditStop(f"would drop below min_remaining={min_remaining}")
        r = requests.get(f"{base}{path}", timeout=25, params={
            "apiKey": key, "regions": regions, "markets": ",".join(markets), "oddsFormat": "american"})
        r.raise_for_status()
        state["spent"] += int(r.headers.get("x-requests-last", cost) or cost)
        rem = r.headers.get("x-requests-remaining")
        state["remaining"] = int(float(rem)) if rem not in (None, "") else state["remaining"]
        return r.json()

    games, props = [], []
    try:
        events = get("/odds", ["spreads", "totals"])
    except _CreditStop as exc:
        logger.warning("Odds API circuit breaker: %s", exc)
        return pd.DataFrame(), pd.DataFrame()
    for ev in events:
        home, away = normalize_team(ev["home_team"]), normalize_team(ev["away_team"])
        totals, spreads = [], []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    if mk["key"] == "totals" and oc.get("name") == "Over":
                        totals.append(oc.get("point"))
                    if mk["key"] == "spreads" and normalize_team(oc.get("name")) == home:
                        spreads.append(oc.get("point"))
        games.append({"event_id": ev["id"], "commence_time": ev.get("commence_time"), "home_team": home,
                      "away_team": away, "total": float(np.median(totals)) if totals else np.nan,
                      "home_spread": float(np.median(spreads)) if spreads else np.nan})
    for g in games:
        try:
            ev = get(f"/events/{g['event_id']}/odds", list(PROP_MARKETS))
        except _CreditStop as exc:
            logger.warning("Odds API circuit breaker: %s (props fetched for %d games)", exc,
                           len({p['event_id'] for p in props}))
            break
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                stat = PROP_MARKETS.get(mk["key"])
                if not stat:
                    continue
                pair: Dict[Tuple[str, float], Dict[str, float]] = {}
                for oc in mk.get("outcomes", []):
                    k = (oc.get("description", ""), oc.get("point"))
                    pair.setdefault(k, {})[oc.get("name", "").lower()] = oc.get("price")
                for (player, line), px in pair.items():
                    if "over" in px and "under" in px and line is not None:
                        props.append({"event_id": g["event_id"], "book": bk["key"], "player": player, "stat": stat,
                                      "line": float(line), "over_odds": px["over"], "under_odds": px["under"]})
    logger.info("Odds API: %d credits spent this run, %s remaining", state["spent"], state["remaining"])
    return pd.DataFrame(games), pd.DataFrame(props)


class _CreditStop(Exception):
    pass


def american_to_prob(odds) -> float:
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def compute_prop_implied_fd_prior(props: pd.DataFrame) -> pd.DataFrame:
    """De-vig over/under prices into market-implied stat means and a partial FanDuel prior.

    mean = line + sd * Phi^-1(p_over_novig), sd = PROP_SD_PER_SQRT[stat] * sqrt(line).
    Medians are taken across books. `vegas_fd_prior` sums only the stats that have markets;
    `n_markets` says how many of the six FanDuel stats are covered.
    """
    cols = ["player_key"] + [f"mkt_{s}" for s in STATS] + ["n_markets", "vegas_fd_prior"]
    if props is None or props.empty:
        return pd.DataFrame(columns=cols)
    p = props.copy()
    p["player_key"] = p["player"].map(normalize_player_name)
    po = p["over_odds"].map(american_to_prob)
    pu = p["under_odds"].map(american_to_prob)
    p["p_over"] = (po / (po + pu)).clip(0.02, 0.98)
    sd = p["stat"].map(PROP_SD_PER_SQRT).fillna(1.0) * np.sqrt(p["line"].clip(lower=0.5))
    p["mkt_mean"] = (p["line"] + sd * norm.ppf(p["p_over"])).clip(lower=0.0)
    wide = p.groupby(["player_key", "stat"])["mkt_mean"].median().unstack("stat")
    wide = wide.reindex(columns=STATS).add_prefix("mkt_")
    wide["n_markets"] = wide.notna().sum(axis=1)
    wide["vegas_fd_prior"] = sum(FD_SCORING[s] * wide[f"mkt_{s}"].fillna(0.0) for s in STATS)
    return wide.reset_index()[cols]


# --------------------------------------------------------------------------------------
# Ownership
# --------------------------------------------------------------------------------------
def _structural_ownership(df: pd.DataFrame) -> pd.Series:
    """Heuristic ownership: points-per-$1k value and raw projection, softmaxed to 900% total
    (9 roster spots x 100%). Not fit to real contest ownership; supply a CSV when you have one."""
    active = (df.get("out_prob", 0.0) < 0.9) & (df["proj_mu"] > 0)
    value = df["proj_mu"] / (df["salary"].clip(lower=3500) / 1000.0)
    z = lambda s: (s - s[active].mean()) / (s[active].std(ddof=0) + 1e-9)
    score = 1.1 * z(value) + 0.6 * z(df["proj_mu"])
    raw = np.exp(score.clip(-6, 6)) * active * (1 - df.get("out_prob", 0.0))
    own = 900.0 * raw / raw.sum()
    for _ in range(5):  # cap and floor, then re-normalise the rest
        own = own.clip(0.3, 60.0)
        own = own * (900.0 / own[active].sum()) if active.any() else own
    return own.where(active, 0.3).clip(0.3, 60.0)


def estimate_slate_ownership(proj_df: pd.DataFrame, ext_ownership_csv: str = "", min_join_rate: float = 0.95) -> pd.Series:
    """Projected ownership % per player: external CSV where it matches, structural heuristic elsewhere."""
    structural = _structural_ownership(proj_df)
    if not ext_ownership_csv:
        return structural
    ext = pd.read_csv(ext_ownership_csv)
    name_c = _col(ext, "player", "name", "Player Name")
    own_c = _col(ext, "proj_own", "ownership", "own", "%own", "projected ownership")
    if not (name_c and own_c):
        raise ValueError(f"{ext_ownership_csv}: need player and ownership columns")
    ext_map = dict(zip(ext[name_c].map(normalize_player_name),
                       pd.to_numeric(ext[own_c].astype(str).str.replace("%", ""), errors="coerce")))
    audit_join_rate(proj_df["player_key"], ext_map.keys(), "ownership CSV", min_join_rate)
    return proj_df["player_key"].map(ext_map).fillna(structural).clip(0.3, 60.0)
