"""FanDuel NBA Classic rules, scoring, and tunable GPP pipeline settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# FanDuel NBA scoring (no 3PM, double-double or triple-double bonuses).
FD_SCORING: Dict[str, float] = {
    "pts": 1.0,
    "reb": 1.2,
    "ast": 1.5,
    "stl": 3.0,
    "blk": 3.0,
    "tov": -1.0,
}

FD_SALARY_CAP: int = 60000

# FanDuel NBA Classic roster: 9 players.
FD_ROSTER_SLOTS: List[Tuple[str, int]] = [("PG", 2), ("SG", 2), ("SF", 2), ("PF", 2), ("C", 1)]
FD_SLOT_ORDER: List[str] = ["PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C"]

# Probability a player does not play, by injury designation (official report or FanDuel indicator).
INJURY_OUT_PROB: Dict[str, float] = {
    "OUT": 1.0,
    "O": 1.0,
    "INJ": 1.0,
    "NA": 1.0,
    "DOUBTFUL": 0.75,
    "D": 0.75,
    "QUESTIONABLE": 0.35,
    "Q": 0.35,
    "GTD": 0.35,
    "PROBABLE": 0.05,
    "P": 0.05,
    "AVAILABLE": 0.0,
    "": 0.0,
}

# Residual SD per sqrt(line) used to turn a de-vigged prop price into a mean.
# pts/reb/ast measured on 2026-27 ESPN holdout residuals (nba-props/results/adv_eval.json);
# stl/blk/tov assume roughly Poisson spread.
PROP_SD_PER_SQRT: Dict[str, float] = {"pts": 1.84, "reb": 1.23, "ast": 1.17, "stl": 1.0, "blk": 1.0, "tov": 1.0}

RIDGE_FEATURES: List[str] = [
    "z_db_dpm",
    "z_db_zts",
    "z_db_opp_dts",
    "z_db_opp_3p_luck",
    "z_db_opp_stop",
    "z_db_opp_drb",
    "z_vegas_implied",
    "z_spread_abs",
    "z_b2b",
    "z_wowy_shift",
]


@dataclass
class GPPConfig:
    # FanDuel rules
    salary_cap: int = FD_SALARY_CAP
    min_salary: int = 56000
    max_per_team: int = 4
    min_games: int = 2

    # Portfolio
    n_sims: int = 2000
    n_lineups: int = 20
    max_exposure: float = 0.45
    min_unique_players: int = 2
    candidate_pool_size: int = 150
    ownership_penalty_lambda: float = 0.8
    dup_penalty_weight: float = 0.5
    random_seed: int = 7

    # Correlation structure (simulator.py)
    rho_same_game_shootout: float = 0.12
    rho_teammate_assist_synergy: float = 0.05
    rho_teammate_usage_cannibal: float = -0.10
    rho_starter_backup_swap: float = -0.25
    rho_opp_bringback_close: float = 0.08

    # Projections
    ridge_features: List[str] = field(default_factory=lambda: list(RIDGE_FEATURES))
    ridge_alpha: float = 200.0
    ridge_adjust_cap: float = 4.0          # max FD points the ridge can move a projection
    shrink_k_minutes: float = 40.0         # position-average prior (minutes) blended into per-minute rates; 240 under-projected stars by ~6 FD pts in the 2026-27 walk-forward
    wowy_padding_minutes: float = 266.0    # Databallr WOWY padding prior
    recent_minutes_weight: float = 0.3     # weight on last-5 minutes vs season minutes (role changes)
    market_weight: float = 0.55            # weight on de-vigged prop means vs structural components
    redistribute_share: float = 0.6        # share of an absent player's minutes handed to teammates (0.6 beat 0.85 in the 2026-27 walk-forward)
    max_minutes: float = 40.0
    sd_floor: float = 3.0
    min_train_dates: int = 7               # walk-forward warm-up
    ridge_min_rows: int = 300              # out-of-sample rows needed before the ridge is fit

    # The Odds API circuit breaker
    odds_api_max_credits_per_run: int = 120
    odds_api_min_remaining_credits: int = 50

    # Entity joins
    min_join_rate: float = 0.95
