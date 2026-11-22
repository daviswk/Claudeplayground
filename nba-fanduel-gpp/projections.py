"""FanDuel projections: structural rate model, market anchoring, ridge correction, walk-forward CV.

Projection for a player tonight:
  1. Per-minute rates for pts/reb/ast/stl/blk/tov over the season, shrunk toward the position
     average with `shrink_k_minutes` of prior (no recency weighting on rates).
  2. Projected minutes = season minutes blended with last-5 minutes (role changes), plus minutes
     freed by absent teammates (same position group first), trimmed for back-to-backs and blowouts.
  3. Matchup multipliers: Vegas implied team total for pts/ast (it already prices pace and the
     opponent's defence); pace and opponent rebounding / ball-security / turnover-forcing for the rest.
  4. Where de-vigged player props exist, each stat is blended toward the market mean.
  5. Databallr WOWY per-36 shifts for absent teammates, when supplied.
  6. A ridge model (alpha=200) on within-slate z-scores, trained on the structural model's own
     out-of-sample residuals from the walk-forward run, nudges the result (capped).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gamma, spearmanr

from config import FD_SCORING, INJURY_OUT_PROB, GPPConfig
from data_sources import STATS, derive_team_factors_from_logs
from normalize import pos_group

logger = logging.getLogger(__name__)

# ridge z-feature -> (raw column, sign)
Z_FEATURES: Dict[str, Tuple[str, float]] = {
    "z_db_dpm": ("dpm", 1.0),
    "z_db_zts": ("zts", 1.0),
    "z_db_opp_dts": ("opp_dts", -1.0),
    "z_db_opp_3p_luck": ("opp_3p_luck", 1.0),
    "z_db_opp_stop": ("opp_stop", 1.0),
    "z_db_opp_drb": ("opp_drb", -1.0),
    "z_vegas_implied": ("vegas_implied", 1.0),
    "z_spread_abs": ("spread_abs", 1.0),
    "z_b2b": ("is_b2b", 1.0),
    "z_wowy_shift": ("wowy_fd_bump", 1.0),
}
DEFAULT_SD_CURVE = (3.0, 0.25)


# --------------------------------------------------------------------------------------
# Ridge
# --------------------------------------------------------------------------------------
class RidgeFDPredictor:
    """Closed-form L2 ridge on residuals (target - anchor). The intercept is fit but not applied
    to new slates, so the model only re-ranks players relative to each other."""

    def __init__(self, features: List[str], alpha: float = 200.0, target: str = "fd_pts", anchor: str = "fd_struct"):
        self.features = list(features)
        self.alpha = float(alpha)
        self.target = target
        self.anchor = anchor
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "RidgeFDPredictor":
        d = df.dropna(subset=[self.target, self.anchor])
        X = d.reindex(columns=self.features).fillna(0.0).to_numpy(float)
        y = (d[self.target] - d[self.anchor]).to_numpy(float)
        self.intercept_ = float(y.mean()) if len(y) else 0.0
        if len(y) == 0:
            self.coef_ = np.zeros(len(self.features))
            return self
        Xc = X - X.mean(axis=0)
        self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(X.shape[1]), Xc.T @ (y - self.intercept_))
        return self

    def predict_adjustment(self, df: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            return np.zeros(len(df))
        return df.reindex(columns=self.features).fillna(0.0).to_numpy(float) @ self.coef_

    def coefficients(self) -> Dict[str, float]:
        return {f: round(float(c), 4) for f, c in zip(self.features, self.coef_ if self.coef_ is not None else [])}


def add_z_features(df: pd.DataFrame, by: str = "date") -> pd.DataFrame:
    """Within-slate z-scores of the ridge inputs (missing raw columns become 0)."""
    out = df.copy()
    for z_col, (raw, sign) in Z_FEATURES.items():
        s = sign * pd.to_numeric(out[raw], errors="coerce").fillna(0.0) if raw in out.columns else pd.Series(0.0, index=out.index)
        g = s.groupby(out[by]) if by in out.columns else s.groupby(np.zeros(len(s)))
        out[z_col] = (s - g.transform("mean")) / (g.transform(lambda x: x.std(ddof=0)) + 1e-6)
    return out


# --------------------------------------------------------------------------------------
# Structural model
# --------------------------------------------------------------------------------------
def player_rates(hist: pd.DataFrame, cfg: GPPConfig) -> pd.DataFrame:
    h = hist.copy()
    h["pos_group"] = h.get("pos", pd.Series("", index=h.index)).map(pos_group)
    agg = h.groupby("player_key").agg(
        min_sum=("min", "sum"), gp=("eid", "nunique"), last_team=("team", "last"),
        first_date=("date", "min"), last_date=("date", "max"), hist_pos_group=("pos_group", "last"),
        **{f"{s}_sum": (s, "sum") for s in STATS if s in h.columns},
    )
    last5 = h.sort_values("date").groupby("player_key").tail(5).groupby("player_key")["min"].mean()
    agg["min_season"] = agg["min_sum"] / agg["gp"]
    agg["min_last5"] = last5
    w = cfg.recent_minutes_weight
    agg["base_min"] = (1 - w) * agg["min_season"] + w * agg["min_last5"]
    prior_tot = h.groupby("pos_group")[["min"] + [s for s in STATS if s in h.columns]].sum()
    for s in STATS:
        if f"{s}_sum" not in agg.columns:
            agg[f"{s}_rate"] = 0.0
            continue
        prior = agg["hist_pos_group"].map(prior_tot[s] / prior_tot["min"]).fillna(prior_tot[s].sum() / prior_tot["min"].sum())
        agg[f"{s}_rate"] = (agg[f"{s}_sum"] + prior * cfg.shrink_k_minutes) / (agg["min_sum"] + cfg.shrink_k_minutes)
    agg["fd_rate"] = sum(FD_SCORING[s] * agg[f"{s}_rate"] for s in STATS)
    # how often the player suits up: games played / team games since his first appearance
    team_dates = h.drop_duplicates(["eid", "team"])[["team", "date"]]
    by_team = {tm: np.sort(g["date"].to_numpy()) for tm, g in team_dates.groupby("team")}
    since = [len(by_team.get(tm, [])) - np.searchsorted(by_team.get(tm, np.array([])), fd)
             for tm, fd in zip(agg["last_team"], agg["first_date"])]
    agg["play_rate"] = (agg["gp"] / np.maximum(np.array(since, dtype=float), 1.0)).clip(0.0, 1.0)
    return agg


def _num_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def _team_context(hist: pd.DataFrame, team_factors: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    tp = hist.groupby(["eid", "team"])["pts"].sum().groupby("team").mean().rename("team_avg_pts")
    ctx = pd.DataFrame(tp)
    if team_factors is not None and not team_factors.empty:
        ctx = ctx.join(team_factors.set_index("team"), how="outer")
    lg_pace = float(ctx["pace"].mean()) if "pace" in ctx and ctx["pace"].notna().any() else np.nan
    return ctx, lg_pace


def _redistribute_minutes(t: pd.DataFrame, absent: pd.DataFrame, cfg: GPPConfig) -> pd.Series:
    """Minutes freed by absent players (x their out probability), 70% to the same position group,
    30% team-wide, proportional to each recipient's base minutes."""
    bump = pd.Series(0.0, index=t.index)
    if absent is None or absent.empty:
        return bump
    freed = (absent["base_min"] * absent["weight"] * cfg.redistribute_share).groupby([absent["team"], absent["pos_group"]]).sum()
    eligible = (t["out_prob"] < 0.5) & t["base_min"].notna() & (t["base_min"] > 0)
    for (team, grp), mins in freed.items():
        if not np.isfinite(mins) or mins <= 0:
            continue
        team_rec = t[eligible & (t["team"] == team)]
        if team_rec.empty:
            continue
        same = team_rec[team_rec["pos_group"] == grp]
        parts = [(same, 0.7), (team_rec, 0.3)] if not same.empty else [(team_rec, 1.0)]
        for rec, share in parts:
            w = rec["base_min"] / rec["base_min"].sum()
            bump.loc[rec.index] += mins * share * w
    return bump


def project_players(
    hist: pd.DataFrame,
    targets: pd.DataFrame,
    team_factors: pd.DataFrame,
    cfg: GPPConfig,
    absent: Optional[pd.DataFrame] = None,
    props: Optional[pd.DataFrame] = None,
    wowy: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Structural FanDuel projection for `targets` (player_key, team, opp, date, out_prob,
    team_spread, vegas_total, optional pos_group) from history strictly before the target date.

    `absent`: player_key, team, weight (= probability they sit) for players whose minutes are freed.
    """
    rates = player_rates(hist, cfg)
    t = targets.copy().reset_index(drop=True)
    t = t.merge(rates[["base_min", "min_season", "gp", "play_rate", "fd_rate", "hist_pos_group", "last_date"] + [f"{s}_rate" for s in STATS]],
                left_on="player_key", right_index=True, how="left")
    if "pos_group" not in t.columns:
        t["pos_group"] = t["hist_pos_group"]
    t["pos_group"] = t["pos_group"].fillna(t["hist_pos_group"]).fillna("F")
    if "out_prob" not in t.columns:
        t["out_prob"] = 0.0

    # absent teammates -> freed minutes
    if absent is not None and not absent.empty:
        a = absent.merge(rates[["base_min", "hist_pos_group"]], left_on="player_key", right_index=True, how="inner")
        a["pos_group"] = a["hist_pos_group"].fillna("F")
        a = a[a["base_min"].notna()]
    else:
        a = pd.DataFrame(columns=["player_key", "team", "weight", "base_min", "pos_group"])
    t["min_bump"] = _redistribute_minutes(t, a, cfg)

    # schedule + blowout
    team_last = hist.groupby("team")["date"].max()
    t["is_b2b"] = (pd.to_datetime(t["date"]) - pd.to_datetime(t["team"].map(team_last))).dt.days.eq(1).astype(int)
    t["spread_abs"] = _num_col(t, "team_spread").abs()
    proj_min = (t["base_min"] + t["min_bump"]).clip(upper=cfg.max_minutes)
    proj_min = proj_min * np.where(t["is_b2b"] == 1, 0.97, 1.0)
    blow = ((t["spread_abs"].fillna(0) - 8).clip(lower=0) * 0.006).where(t["base_min"] >= 28, 0.0)
    t["proj_min"] = (proj_min * (1 - blow)).fillna(0.0)
    # a team only has 240 minutes: scale down when expected minutes (x chance of playing) exceed it
    t["p_play"] = (1 - t["out_prob"]) * t["play_rate"].fillna(0.0)
    exp_team = (t["proj_min"] * t["p_play"]).groupby(t["team"]).transform("sum")
    t["min_scale"] = np.where(exp_team > 240.0, 240.0 / exp_team.where(exp_team > 0, 1.0), 1.0)
    t["proj_min"] = t["proj_min"] * t["min_scale"]

    # matchup multipliers
    ctx, lg_pace = _team_context(hist, team_factors)
    opp = ctx.reindex(t["opp"]).reset_index(drop=True)
    own = ctx.reindex(t["team"]).reset_index(drop=True)
    for c in ["dts", "three_p_luck", "stop", "drb", "otov", "dtov", "pace"]:
        t[f"opp_{c}"] = opp[c].to_numpy() if c in opp.columns else np.nan
    pace_f = (t["opp_pace"] / lg_pace).clip(0.92, 1.08).fillna(1.0) if np.isfinite(lg_pace) else pd.Series(1.0, index=t.index)
    t["vegas_implied"] = _num_col(t, "vegas_total") / 2 - _num_col(t, "team_spread") / 2
    team_avg = own["team_avg_pts"].to_numpy() if "team_avg_pts" in own.columns else np.nan
    vegas_f = (t["vegas_implied"] / team_avg).clip(0.85, 1.15)
    def_f = (1 - 0.5 * t["opp_dts"].fillna(0) / 57.0) * pace_f
    scoring_f = vegas_f.fillna(def_f).clip(0.85, 1.15)
    mult = {
        "pts": scoring_f,
        "ast": scoring_f ** 0.7,
        "reb": (pace_f * (1 - 0.25 * t["opp_drb"].fillna(0) / 27.0)).clip(0.85, 1.15),
        "stl": (pace_f * (1 - 0.5 * t["opp_otov"].fillna(0) / 13.0)).clip(0.85, 1.15),
        "blk": pace_f,
        "tov": (pace_f * (1 + 0.5 * t["opp_dtov"].fillna(0) / 13.0)).clip(0.85, 1.15),
    }
    for s in STATS:
        t[f"proj_{s}"] = t[f"{s}_rate"] * t["proj_min"] * mult[s]

    # market anchoring per stat
    t["n_markets"] = 0
    if props is not None and not props.empty:
        p = props.set_index("player_key")
        for s in STATS:
            col = f"mkt_{s}"
            if col in p.columns:
                m = t["player_key"].map(p[col])
                has = m.notna() & t[f"proj_{s}"].notna()
                t.loc[has, f"proj_{s}"] = cfg.market_weight * m[has] + (1 - cfg.market_weight) * t.loc[has, f"proj_{s}"]
                t["n_markets"] += has.astype(int)

    # WOWY per-36 shifts for absent teammates
    t["wowy_fd_bump"] = 0.0
    if wowy is not None and not wowy.empty and not a.empty:
        wts = a.set_index("player_key")["weight"]
        w = wowy[wowy["teammate_out_key"].isin(wts.index)].copy()
        w["fd_shift"] = w["fd_p36_shift_shrunk"] * w["teammate_out_key"].map(wts)
        per_player = w.groupby("player_key")["fd_shift"].sum()
        t["wowy_fd_bump"] = t["player_key"].map(per_player).fillna(0.0) * t["proj_min"] / 36.0

    t["fd_struct"] = sum(FD_SCORING[s] * t[f"proj_{s}"] for s in STATS) + t["wowy_fd_bump"]
    t["has_history"] = t["base_min"].notna()
    return t


# --------------------------------------------------------------------------------------
# Walk-forward CV
# --------------------------------------------------------------------------------------
def _absent_rotation(hist: pd.DataFrame, test: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Rotation players (>=12 min, played in one of the team's last 3 games) missing from tonight's box."""
    played = set(zip(test["team"], test["player_key"]))
    rows = []
    for team in test["team"].unique():
        th = hist[hist["team"] == team]
        last3 = th.drop_duplicates("eid").sort_values("date")["eid"].tail(3)
        recent = set(th[th["eid"].isin(last3)]["player_key"])
        for pk in recent:
            if (team, pk) in played or pk not in rates.index:
                continue
            if rates.at[pk, "base_min"] >= 12 and rates.at[pk, "last_team"] == team:
                rows.append({"player_key": pk, "team": team, "weight": 1.0})
    return pd.DataFrame(rows, columns=["player_key", "team", "weight"])


def _metrics(pred: np.ndarray, actual: np.ndarray, dates: np.ndarray) -> Dict[str, float]:
    err = pred - actual
    rs, cap = [], []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 10:
            continue
        rs.append(spearmanr(pred[m], actual[m]).correlation)
        k = max(1, int(round(0.1 * m.sum())))
        top_a = set(np.argsort(-actual[m])[:k])
        top_p = set(np.argsort(-pred[m])[:k])
        cap.append(len(top_a & top_p) / k)
    return {"mae": round(float(np.mean(np.abs(err))), 3), "rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
            "bias": round(float(np.mean(err)), 3), "spearman_r": round(float(np.nanmean(rs)), 3),
            "top10pct_capture": round(float(np.mean(cap)), 3), "n": int(len(err))}


def estimate_sd_curve(oos: pd.DataFrame, pred_col: str = "fd_struct") -> Tuple[float, float]:
    """sd(FD points) ~ a + b * projection, fit on out-of-sample residuals by projection decile."""
    d = oos.dropna(subset=[pred_col, "fd_pts"])
    d = d[d[pred_col] > 1]
    if len(d) < 200:
        return DEFAULT_SD_CURVE
    bins = pd.qcut(d[pred_col], 10, duplicates="drop")
    g = d.groupby(bins, observed=True).apply(lambda x: pd.Series({"mu": x[pred_col].mean(), "sd": (x["fd_pts"] - x[pred_col]).std()}))
    b, a = np.polyfit(g["mu"], g["sd"], 1)
    return float(max(a, 1.0)), float(np.clip(b, 0.05, 0.6))


def walk_forward_backtest(
    player_logs: pd.DataFrame,
    db_teams: Optional[pd.DataFrame] = None,
    db_players: Optional[pd.DataFrame] = None,
    cfg: Optional[GPPConfig] = None,
    return_oos: bool = False,
    min_train_dates: Optional[int] = None,
):
    """Expanding-window walk-forward: for each date, project every player who played from earlier
    games only (absent rotation players treated as the injury report), and compare
      - season_avg_baseline: player's FanDuel average to date
      - structural: project_players
      - structural_plus_ridge: structural + ridge fit on earlier dates' out-of-sample residuals
    Team factors are re-derived from prior games when the logs carry shooting splits (no look-ahead);
    otherwise the static `db_teams` table is used and flagged.
    """
    cfg = cfg or GPPConfig()
    logs = player_logs.copy()
    if logs.empty:
        return ({}, pd.DataFrame()) if return_oos else {}
    derivable = {"fga", "fta", "oreb", "dreb", "tpa", "stl", "blk"}.issubset(logs.columns)
    warmup = cfg.min_train_dates if min_train_dates is None else int(min_train_dates)
    dates = sorted(logs["date"].unique())
    oos_parts: List[pd.DataFrame] = []
    for i, d in enumerate(dates):
        if i < warmup:
            continue
        hist = logs[logs["date"] < d]
        test = logs[logs["date"] == d].copy()
        tf = derive_team_factors_from_logs(hist) if derivable else (db_teams if db_teams is not None else pd.DataFrame())
        rates = player_rates(hist, cfg)
        absent = _absent_rotation(hist, test, rates)
        targets = test[["eid", "date", "player_key", "team", "opp", "fd_pts", "min"]].copy()
        targets["pos_group"] = test["pos"].map(pos_group) if "pos" in test else "F"
        targets["team_spread"] = pd.to_numeric(test.get("spread"), errors="coerce")
        targets["vegas_total"] = pd.to_numeric(test.get("total"), errors="coerce")
        targets["out_prob"] = 0.0
        proj = project_players(hist, targets, tf, cfg, absent=absent)
        proj["baseline"] = proj["player_key"].map(hist.groupby("player_key")["fd_pts"].mean())
        oos_parts.append(proj)
    if not oos_parts:
        return ({}, pd.DataFrame()) if return_oos else {}
    oos = pd.concat(oos_parts, ignore_index=True)
    oos = oos[oos["has_history"] & oos["baseline"].notna()].copy()
    if db_players is not None and not db_players.empty:
        oos = oos.merge(db_players[["player_key", "dpm", "zts"]], on="player_key", how="left")
    oos = add_z_features(oos, by="date")

    # ridge, refit before each date on earlier out-of-sample rows only
    oos["ridge_adj"] = np.nan
    ridge_dates = 0
    for d in sorted(oos["date"].unique()):
        train = oos[oos["date"] < d]
        if len(train) < cfg.ridge_min_rows:
            continue
        model = RidgeFDPredictor(cfg.ridge_features, alpha=cfg.ridge_alpha).fit(train)
        m = oos["date"] == d
        oos.loc[m, "ridge_adj"] = np.clip(model.predict_adjustment(oos[m]), -cfg.ridge_adjust_cap, cfg.ridge_adjust_cap)
        ridge_dates += 1
    oos["fd_ridge"] = oos["fd_struct"] + oos["ridge_adj"].fillna(0.0)

    a, dts = oos["fd_pts"].to_numpy(float), oos["date"].to_numpy()
    all_dates = {
        "season_avg_baseline": _metrics(oos["baseline"].to_numpy(float), a, dts),
        "structural": _metrics(oos["fd_struct"].to_numpy(float), a, dts),
        # ridge adjustment is 0 on warm-up dates before it has enough training rows
        "structural_plus_ridge": _metrics(oos["fd_ridge"].to_numpy(float), a, dts),
    }
    metrics = {
        "n_eval_dates": int(oos["date"].nunique()),
        "n_eval_rows": int(len(oos)),
        "team_factors": "espn_derived_from_prior_games" if derivable else "static_input_table (look-ahead)",
    }
    for key, name in [("baseline", "season_avg_baseline"), ("structural", "structural"), ("ridge", "structural_plus_ridge")]:
        for m in ("mae", "rmse", "bias", "spearman_r", "top10pct_capture"):
            metrics[f"{key}_{m.replace('_r', '') if m == 'spearman_r' else m}"] = all_dates[name][m]
    metrics["all_dates"] = all_dates
    r = oos["ridge_adj"].notna().to_numpy()
    if r.any():
        metrics["ridge_dates"] = {
            "n_dates": ridge_dates,
            "season_avg_baseline": _metrics(oos["baseline"].to_numpy(float)[r], a[r], dts[r]),
            "structural": _metrics(oos["fd_struct"].to_numpy(float)[r], a[r], dts[r]),
            "structural_plus_ridge": _metrics(oos["fd_ridge"].to_numpy(float)[r], a[r], dts[r]),
        }
    metrics["sd_curve"] = list(estimate_sd_curve(oos))
    return (metrics, oos) if return_oos else metrics


def fit_ridge_on_oos(oos: pd.DataFrame, cfg: GPPConfig) -> Optional[RidgeFDPredictor]:
    if oos is None or len(oos) < cfg.ridge_min_rows:
        logger.info("Ridge skipped: %d out-of-sample rows (< %d).", 0 if oos is None else len(oos), cfg.ridge_min_rows)
        return None
    model = RidgeFDPredictor(cfg.ridge_features, alpha=cfg.ridge_alpha).fit(oos)
    logger.info("Ridge coefficients (FD pts per 1 sd): %s", model.coefficients())
    return model


# --------------------------------------------------------------------------------------
# Tonight's slate
# --------------------------------------------------------------------------------------
def _vegas_by_team(vegas_games_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if vegas_games_df is None or vegas_games_df.empty:
        return pd.DataFrame(columns=["team", "vegas_total", "team_spread"])
    g = vegas_games_df
    home = pd.DataFrame({"team": g["home_team"], "vegas_total": g["total"], "team_spread": g["home_spread"]})
    away = pd.DataFrame({"team": g["away_team"], "vegas_total": g["total"], "team_spread": -pd.to_numeric(g["home_spread"], errors="coerce")})
    return pd.concat([home, away]).drop_duplicates("team", keep="last")


def build_slate_projections(
    slate_df: pd.DataFrame,
    player_logs: pd.DataFrame,
    databallr_teams: Optional[pd.DataFrame] = None,
    databallr_players: Optional[pd.DataFrame] = None,
    databallr_wowy: Optional[pd.DataFrame] = None,
    injuries_df: Optional[pd.DataFrame] = None,
    vegas_games_df: Optional[pd.DataFrame] = None,
    vegas_props_prior: Optional[pd.DataFrame] = None,
    ridge_model: Optional[RidgeFDPredictor] = None,
    cfg: Optional[GPPConfig] = None,
    slate_date: Optional[str] = None,
    sd_curve: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Per-player FanDuel projections for the slate: proj_min, proj_mu, proj_sd, proj_p90/p99, out_prob."""
    cfg = cfg or GPPConfig()
    df = slate_df.copy().reset_index(drop=True)
    logs = player_logs if player_logs is not None else pd.DataFrame()
    if not slate_date:
        slate_date = ((pd.to_datetime(logs["date"]).max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                      if not logs.empty else pd.Timestamp.today().strftime("%Y-%m-%d"))
    hist = logs[logs["date"] < slate_date] if not logs.empty else logs
    team_factors = databallr_teams if databallr_teams is not None and not databallr_teams.empty else derive_team_factors_from_logs(hist)

    # injuries: worst of FanDuel indicator and official report
    fd_prob = df.get("fd_injury", pd.Series("", index=df.index)).fillna("").map(lambda x: INJURY_OUT_PROB.get(str(x).upper(), 0.0))
    off_prob = pd.Series(0.0, index=df.index)
    if injuries_df is not None and not injuries_df.empty:
        off_prob = df["player_key"].map(injuries_df.drop_duplicates("player_key").set_index("player_key")["out_prob"]).fillna(0.0)
    df["out_prob"] = np.maximum(fd_prob.to_numpy(float), off_prob.to_numpy(float))

    df = df.merge(_vegas_by_team(vegas_games_df), on="team", how="left")
    df["date"] = slate_date
    df["pos_group"] = df["primary_pos"].map(pos_group)

    if hist is None or hist.empty:
        df["fd_struct"] = df.get("fppg", 0.0)
        df["has_history"] = False
        df["proj_min"] = 0.0
        out = df
    else:
        absent = df.loc[df["out_prob"] > 0, ["player_key", "team"]].assign(weight=df.loc[df["out_prob"] > 0, "out_prob"])
        out = project_players(hist, df, team_factors, cfg, absent=absent, props=vegas_props_prior, wowy=databallr_wowy)

    if databallr_players is not None and not databallr_players.empty:
        out = out.merge(databallr_players[["player_key", "dpm", "zts"]].drop_duplicates("player_key"), on="player_key", how="left")
    out = add_z_features(out, by="date")
    out["ridge_adj"] = 0.0
    if ridge_model is not None:
        out["ridge_adj"] = np.clip(ridge_model.predict_adjustment(out), -cfg.ridge_adjust_cap, cfg.ridge_adjust_cap)

    # players with no box-score history fall back to FanDuel's FPPG
    fppg = pd.to_numeric(out.get("fppg", 0.0), errors="coerce")
    no_hist = ~out["has_history"].astype(bool)
    out["proj_source"] = np.where(no_hist, "fppg_fallback", "structural")
    out["proj_mu"] = np.where(no_hist, fppg, out["fd_struct"] + out["ridge_adj"])
    out["proj_mu"] = pd.to_numeric(out["proj_mu"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out.loc[no_hist, "proj_min"] = np.where(out.loc[no_hist, "proj_mu"] > 0, 20.0, 0.0)

    a, b = sd_curve or DEFAULT_SD_CURVE
    out["proj_sd"] = np.maximum(cfg.sd_floor, a + b * out["proj_mu"])
    k = np.maximum((out["proj_mu"] / out["proj_sd"]) ** 2, 1.5)
    theta = out["proj_sd"] ** 2 / out["proj_mu"].clip(lower=0.1)
    out["proj_p90"] = gamma.ppf(0.90, k, scale=theta)
    out["proj_p99"] = gamma.ppf(0.99, k, scale=theta)
    out["spread_abs"] = out["spread_abs"].fillna(5.0) if "spread_abs" in out else 5.0
    out["vegas_total"] = pd.to_numeric(out.get("vegas_total"), errors="coerce").fillna(226.0)
    out["value"] = out["proj_mu"] / (out["salary"] / 1000.0)
    return out
