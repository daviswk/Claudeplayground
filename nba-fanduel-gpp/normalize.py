"""Canonical entity crosswalk: player names, team codes, FanDuel positions, join-rate audits."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable, List, Tuple

logger = logging.getLogger(__name__)

TEAM_FULL_NAMES = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
_FULL_UPPER = {k.upper(): v for k, v in TEAM_FULL_NAMES.items()}

TEAM_ALIASES = {
    "GS": "GSW", "GOS": "GSW", "SA": "SAS", "SAN": "SAS", "NY": "NYK", "NO": "NOP", "NOR": "NOP", "NOH": "NOP",
    "UTAH": "UTA", "UTH": "UTA", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "BK": "BKN", "NJ": "BKN", "NJN": "BKN",
    "CHO": "CHA", "CHH": "CHA", "LAK": "LAL",
}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
FD_SLOTS = ("PG", "SG", "SF", "PF", "C")
_GENERIC_POS = {"G": ["PG", "SG"], "F": ["SF", "PF"], "GF": ["SG", "SF"], "FC": ["PF", "C"]}


def normalize_team(code) -> str:
    """Map any team code or full name (ESPN, FanDuel, Odds API, NBA PDF) to a canonical 3-letter code."""
    if code is None:
        return ""
    s = str(code).strip()
    if not s or s.lower() == "nan":
        return ""
    up = s.upper()
    if up in _FULL_UPPER:
        return _FULL_UPPER[up]
    return TEAM_ALIASES.get(up, up)


def normalize_game(game: str) -> str:
    """'GS@SA' -> 'GSW@SAS'."""
    if not game or "@" not in str(game):
        return str(game or "")
    away, home = str(game).split("@", 1)
    return f"{normalize_team(away)}@{normalize_team(home)}"


def normalize_player_name(name) -> str:
    """Join key for players: ASCII, lowercase, no punctuation or suffixes; 'Last, First' is flipped."""
    if name is None:
        return ""
    s = str(name).strip()
    if not s or s.lower() == "nan":
        return ""
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace(".", "").replace("'", "").replace("`", "")
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    tokens = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def parse_fd_positions(pos_raw) -> Tuple[str, ...]:
    """'PG/SG' -> ('PG', 'SG'); generic 'G'/'F' expand to both FanDuel slots."""
    if pos_raw is None:
        return ()
    out: List[str] = []
    for tok in re.split(r"[/,\- ]+", str(pos_raw).upper()):
        tok = tok.strip()
        if not tok:
            continue
        for slot in _GENERIC_POS.get(tok, [tok]):
            if slot in FD_SLOTS and slot not in out:
                out.append(slot)
    return tuple(out)


def pos_group(pos) -> str:
    """Collapse any position label to G / F / C for priors and minutes redistribution."""
    p = str(pos or "").upper()
    if not p or p == "NAN":
        return "F"
    first = re.split(r"[/\- ]", p)[0]
    if first in ("PG", "SG", "G", "GF"):
        return "G"
    if first == "C":
        return "C"
    return "F"


def audit_join_rate(left_keys: Iterable[str], right_keys: Iterable[str], label: str = "",
                    min_rate: float = 0.95, strict: bool = False) -> Tuple[float, List[str]]:
    """Share of left keys found in right keys, plus the unmatched keys.
    Logs a warning below `min_rate`; with `strict`, raises ValueError instead."""
    left = [k for k in left_keys if k]
    right = set(k for k in right_keys if k)
    if not left:
        return 1.0, []
    unmatched = sorted(set(k for k in left if k not in right))
    rate = sum(k in right for k in left) / len(left)
    if rate < min_rate:
        msg = f"Join audit {label}: {100 * rate:.1f}% matched (< {100 * min_rate:.0f}%). Unmatched sample: {unmatched[:15]}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    else:
        logger.info("Join audit %s: %.1f%% matched.", label, 100 * rate)
    return rate, unmatched
