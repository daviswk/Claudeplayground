"""HiGHS MILP FanDuel Classic NBA GPP portfolio optimizer and FanDuel CSV exporter."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from config import FD_ROSTER_SLOTS, FD_SLOT_ORDER, GPPConfig

logger = logging.getLogger(__name__)


def solve_single_fanduel_lineup(
    slate_df: pd.DataFrame,
    scores: np.ndarray,
    cfg: Optional[GPPConfig] = None,
    locked_indices: Optional[List[int]] = None,
    excluded_indices: Optional[List[int]] = None,
) -> Optional[Dict]:
    """Solve the exact FanDuel NBA Classic MILP via HiGHS (`scipy.optimize.milp`).

    Decision variables:
      - x_{i, s} in {0, 1} for each player i and each FanDuel-eligible slot s in {PG, SG, SF, PF, C}.
      - y_i = sum_s x_{i, s} <= 1 (each player used at most once).
    Constraints:
      - Exact FanDuel roster slots: 2 PG, 2 SG, 2 SF, 2 PF, 1 C (9 players).
      - Salary cap: min_salary <= sum_i salary_i * y_i <= salary_cap ($60,000).
      - Max 4 players from any single NBA team.
      - At least 2 distinct games represented (enforced via max 8 players from any single game).
    """
    cfg = cfg or GPPConfig()
    n = len(slate_df)
    if n < 9:
        return None

    positions_list = slate_df["positions"].tolist()
    salaries = slate_df["salary"].to_numpy(dtype=float)
    teams = slate_df["team"].to_numpy()
    games = slate_df["game"].to_numpy()
    out_probs = slate_df.get("out_prob", pd.Series(0.0, index=slate_df.index)).to_numpy(dtype=float)

    excluded = set(excluded_indices or [])
    locked = set(locked_indices or [])
    for i in range(n):
        if out_probs[i] >= 0.90 or not positions_list[i]:
            excluded.add(i)

    # Map each (player_idx, slot) pair to a binary variable index
    var_map: List[Tuple[int, str]] = []
    player_to_vars: Dict[int, List[int]] = {i: [] for i in range(n)}
    slot_to_vars: Dict[str, List[int]] = {slot: [] for slot, _ in FD_ROSTER_SLOTS}

    for i in range(n):
        if i in excluded:
            continue
        for slot in positions_list[i]:
            if slot in slot_to_vars:
                v_idx = len(var_map)
                var_map.append((i, slot))
                player_to_vars[i].append(v_idx)
                slot_to_vars[slot].append(v_idx)

    n_vars = len(var_map)
    if n_vars < 9:
        return None

    # Objective: maximize sum score_i * x_{i, s} -> minimize -score
    c = np.array([-float(scores[i]) for i, _ in var_map], dtype=float)

    rows_A = []
    lb = []
    ub = []

    # 1. Exact slot counts: 2 PG, 2 SG, 2 SF, 2 PF, 1 C
    for slot, req_count in FD_ROSTER_SLOTS:
        row = np.zeros(n_vars, dtype=float)
        if len(slot_to_vars[slot]) < req_count:
            return None
        row[slot_to_vars[slot]] = 1.0
        rows_A.append(row)
        lb.append(float(req_count))
        ub.append(float(req_count))

    # 2. Each player used at most once (or exactly once if locked)
    for i, v_idxs in player_to_vars.items():
        if not v_idxs:
            if i in locked:
                return None
            continue
        row = np.zeros(n_vars, dtype=float)
        row[v_idxs] = 1.0
        rows_A.append(row)
        lb.append(1.0 if i in locked else 0.0)
        ub.append(1.0)

    # 3. Salary cap constraint
    sal_row = np.array([salaries[i] for i, _ in var_map], dtype=float)
    rows_A.append(sal_row)
    lb.append(float(cfg.min_salary))
    ub.append(float(cfg.salary_cap))

    # 4. Max 4 players per team
    for tm in np.unique(teams):
        tm_vars = [v_idx for v_idx, (i, _) in enumerate(var_map) if teams[i] == tm]
        if len(tm_vars) > cfg.max_per_team:
            row = np.zeros(n_vars, dtype=float)
            row[tm_vars] = 1.0
            rows_A.append(row)
            lb.append(0.0)
            ub.append(float(cfg.max_per_team))

    # 5. At least 2 distinct games -> at most 8 players from any single game
    unique_games = np.unique(games)
    if len(unique_games) >= cfg.min_games:
        for gm in unique_games:
            gm_vars = [v_idx for v_idx, (i, _) in enumerate(var_map) if games[i] == gm]
            if len(gm_vars) >= 9:
                row = np.zeros(n_vars, dtype=float)
                row[gm_vars] = 1.0
                rows_A.append(row)
                lb.append(0.0)
                ub.append(8.0)

    A = np.vstack(rows_A)
    res = milp(
        c=c,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(A, lb, ub),
    )
    if not res.success or res.x is None:
        return None

    chosen = [var_map[v] for v in range(n_vars) if res.x[v] > 0.5]
    if len(chosen) != 9:
        return None

    # Order slots strictly to match FanDuel CSV: PG, PG, SG, SG, SF, SF, PF, PF, C
    by_slot: Dict[str, List[int]] = {s: [] for s, _ in FD_ROSTER_SLOTS}
    for p_idx, slot in chosen:
        by_slot[slot].append(p_idx)
    for slot in by_slot:
        by_slot[slot].sort(key=lambda idx: -salaries[idx])

    ordered_indices: List[int] = []
    slot_cursor: Dict[str, int] = {s: 0 for s, _ in FD_ROSTER_SLOTS}
    for slot in FD_SLOT_ORDER:
        k = slot_cursor[slot]
        ordered_indices.append(by_slot[slot][k])
        slot_cursor[slot] = k + 1

    return {
        "player_indices": ordered_indices,
        "player_set": frozenset(ordered_indices),
        "salary": int(sum(salaries[i] for i in ordered_indices)),
    }


def _simulate_field_benchmarks(
    slate_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    cfg: GPPConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-sim Top-1% and Top-0.1% GPP field winning thresholds.

    Generates ownership-weighted chalk field lineups so each simulated slate has
    a realistic tournament cash/top-1%/top-0.1% bar.
    """
    n_sims, n_players = sim_matrix.shape
    own = slate_df["proj_own"].to_numpy(dtype=float).clip(0.3, 60.0)
    rng = np.random.default_rng(cfg.random_seed + 999)

    # Generate 40 ownership-biased field archetypes to represent the public field
    field_lineups = []
    seen = set()
    for _ in range(60):
        noise = rng.normal(0.0, 4.5, size=n_players)
        chalk_score = slate_df["proj_mu"].to_numpy(dtype=float) + 2.5 * np.log(own / 5.0) + noise
        lu = solve_single_fanduel_lineup(slate_df, chalk_score, cfg=cfg)
        if lu and lu["player_set"] not in seen:
            seen.add(lu["player_set"])
            field_lineups.append(lu["player_indices"])
        if len(field_lineups) >= 35:
            break

    if not field_lineups:
        # Fallback: top-9 average scaled
        top_scores = np.sort(sim_matrix, axis=1)[:, -9:].sum(axis=1)
        return top_scores * 0.86, top_scores * 0.93

    field_scores = np.column_stack([sim_matrix[:, idxs].sum(axis=1) for idxs in field_lineups])
    # Top-1% and Top-0.1% bars estimated from the upper tail of the public field distribution
    field_mean = field_scores.mean(axis=1)
    field_sd = field_scores.std(axis=1) + 1e-6
    top_1pct_bar = np.maximum(np.quantile(field_scores, 0.92, axis=1), field_mean + 2.05 * field_sd)
    top_01pct_bar = np.maximum(np.quantile(field_scores, 0.98, axis=1), field_mean + 2.85 * field_sd)
    return top_1pct_bar, top_01pct_bar


def _score_candidate(c: Dict, sim_matrix: np.ndarray, top_1pct_bar: np.ndarray, top_01pct_bar: np.ndarray,
                     own: np.ndarray, cfg: GPPConfig) -> None:
    """Attach simulated score distribution, top-1% / top-0.1% hit rates and duplication-adjusted GPP EV."""
    idxs = c["player_indices"]
    scores = sim_matrix[:, idxs].sum(axis=1)
    # Hyper-skewed FanDuel GPP payout curve: Top-1% = 4x, Top-0.1% = 40x
    hit_1pct = (scores >= top_1pct_bar).astype(float)
    hit_01pct = (scores >= top_01pct_bar).astype(float)
    lu_own = own[idxs]
    geo_own = float(np.exp(np.mean(np.log(lu_own))))
    # Higher geometric ownership -> higher split-pot duplication risk in large GPPs
    dup_factor = 1.0 / (1.0 + cfg.dup_penalty_weight * max(geo_own - 12.0, 0.0) / 10.0)
    adj_payouts = (4.0 * hit_1pct + 36.0 * hit_01pct) * dup_factor
    c["mean_pts"] = float(scores.mean())
    c["p90_pts"] = float(np.quantile(scores, 0.90))
    c["p99_pts"] = float(np.quantile(scores, 0.99))
    c["top_1pct_rate"] = float(hit_1pct.mean())
    c["top_01pct_rate"] = float(hit_01pct.mean())
    c["sum_own"] = round(float(lu_own.sum()), 1)
    c["geo_own"] = round(geo_own, 2)
    c["gpp_ev"] = float(adj_payouts.mean())
    c["adj_payouts"] = adj_payouts


def build_gpp_portfolio(
    slate_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    cfg: Optional[GPPConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate and select an optimal N-lineup FanDuel NBA Classic GPP portfolio.

    Pipeline:
      1. Solve independent correlated Monte Carlo draw realizations via HiGHS MILP
         with mild ownership-leverage adjustment (`-lambda * log(pOwn)`) to build
         a diverse candidate pool (no sequential integer-cut greediness).
      2. Score every candidate across all `n_sims` slates against simulated field
         Top-1% / Top-0.1% benchmarks to compute Tournament GPP EV & duplication risk.
      3. Select the final `n_lineups` portfolio via greedy marginal GPP EV maximization
         subject to `max_exposure` and `min_unique_players`.

    Returns:
      - `fanduel_upload_df`: Exact FanDuel upload CSV (`PG,PG,SG,SG,SF,SF,PF,PF,C`).
      - `portfolio_summary_df`: Lineup-level diagnostics (proj, p90, p99, top1%, top0.1%, EV, own, stack).
      - `exposure_df`: Player-level exposure vs. projected ownership (leverage).
    """
    cfg = cfg or GPPConfig()
    n_sims, n_players = sim_matrix.shape
    own = slate_df["proj_own"].to_numpy(dtype=float).clip(0.3, 60.0)
    log_own_pen = cfg.ownership_penalty_lambda * np.log(own / 10.0)

    rng = np.random.default_rng(cfg.random_seed + 1)
    draw_indices = rng.choice(n_sims, size=min(cfg.candidate_pool_size, n_sims), replace=False)

    candidates: List[Dict] = []
    seen_sets = set()

    for idx_i, sim_idx in enumerate(draw_indices):
        # Vary leverage weight smoothly across draws to cover balanced + contrarian builds
        lam_scale = 0.5 + 1.0 * (idx_i / max(len(draw_indices) - 1, 1))
        obj_scores = sim_matrix[sim_idx] - lam_scale * log_own_pen
        lu = solve_single_fanduel_lineup(slate_df, obj_scores, cfg=cfg)
        if lu and lu["player_set"] not in seen_sets:
            seen_sets.add(lu["player_set"])
            candidates.append(lu)

    if not candidates:
        raise RuntimeError("HiGHS MILP could not find any feasible FanDuel lineup for this slate.")

    # Evaluate every candidate across all simulated slates
    top_1pct_bar, top_01pct_bar = _simulate_field_benchmarks(slate_df, sim_matrix, cfg)
    for c in candidates:
        _score_candidate(c, sim_matrix, top_1pct_bar, top_01pct_bar, own, cfg)

    # Greedy portfolio selection maximizing marginal portfolio best-lineup EV
    # subject to max_exposure and min_unique_players
    selected: List[Dict] = []
    max_player_count = max(1, int(np.floor(cfg.max_exposure * cfg.n_lineups + 1e-9)))
    player_counts = np.zeros(n_players, dtype=int)
    portfolio_best_payout = np.zeros(n_sims, dtype=float)

    remaining = list(range(len(candidates)))
    while len(selected) < cfg.n_lineups:
        best_j = None
        best_gain = -1e18
        for j in remaining:
            cand = candidates[j]
            # Check min_unique against all already selected lineups
            if any(len(cand["player_set"] - s["player_set"]) < cfg.min_unique_players for s in selected):
                continue
            # Check max_exposure
            if any(player_counts[p] >= max_player_count for p in cand["player_indices"]):
                continue
            # Marginal gain in portfolio expected best-lineup payout + small standalone EV tie-breaker
            new_best = np.maximum(portfolio_best_payout, cand["adj_payouts"])
            marginal_gain = float((new_best - portfolio_best_payout).mean()) + 0.15 * cand["gpp_ev"]
            if marginal_gain > best_gain:
                best_gain = marginal_gain
                best_j = j

        if best_j is None:
            # Pool exhausted: solve fresh draws with exposure-capped players excluded, so the
            # max_exposure and min_unique_players limits are never relaxed.
            capped = [p for p in range(n_players) if player_counts[p] >= max_player_count]
            spread = float(np.std(sim_matrix.mean(axis=0))) + 1.0
            for _attempt in range(40):
                sim_idx = int(rng.integers(n_sims))
                diversity_pen = (0.25 + 0.15 * _attempt) * spread * player_counts
                obj = sim_matrix[sim_idx] - log_own_pen - diversity_pen
                lu = solve_single_fanduel_lineup(slate_df, obj, cfg=cfg, excluded_indices=capped)
                if lu is None or lu["player_set"] in seen_sets:
                    continue
                seen_sets.add(lu["player_set"])
                if any(len(lu["player_set"] - s["player_set"]) < cfg.min_unique_players for s in selected):
                    continue
                _score_candidate(lu, sim_matrix, top_1pct_bar, top_01pct_bar, own, cfg)
                candidates.append(lu)
                remaining.append(len(candidates) - 1)
                best_j = len(candidates) - 1
                break
        if best_j is None:
            logger.warning("Stopped at %d lineups: no feasible lineup left under max_exposure/min_unique.", len(selected))
            break

        chosen = candidates[best_j]
        selected.append(chosen)
        remaining.remove(best_j)
        portfolio_best_payout = np.maximum(portfolio_best_payout, chosen["adj_payouts"])
        for p in chosen["player_indices"]:
            player_counts[p] += 1

    # Format FanDuel upload CSV & portfolio diagnostics
    upload_rows = []
    summary_rows = []
    fd_ids = slate_df["fd_id"].tolist()
    names = slate_df["name"].tolist()
    games = slate_df["game"].tolist()
    teams = slate_df["team"].tolist()

    for lu_num, lu in enumerate(selected, start=1):
        idxs = lu["player_indices"]
        upload_rows.append([f"{fd_ids[i]}:{names[i]}" for i in idxs])
        game_counts = pd.Series([games[i] for i in idxs]).value_counts()
        top_game = f"{game_counts.index[0]} ({int(game_counts.iloc[0])})"
        summary_rows.append(
            {
                "lineup_id": lu_num,
                "salary": lu["salary"],
                "mean_pts": round(lu["mean_pts"], 2),
                "p90_pts": round(lu["p90_pts"], 2),
                "p99_pts": round(lu["p99_pts"], 2),
                "top_1pct_rate": round(lu["top_1pct_rate"] * 100, 2),
                "top_01pct_rate": round(lu["top_01pct_rate"] * 100, 2),
                "gpp_ev": round(lu["gpp_ev"], 3),
                "sum_own_pct": lu["sum_own"],
                "geo_own_pct": lu["geo_own"],
                "primary_game_stack": top_game,
                "n_games": int(len(game_counts)),
                "players": " | ".join(f"{FD_SLOT_ORDER[k]}:{names[i]}({teams[i]})" for k, i in enumerate(idxs)),
            }
        )

    fanduel_upload_df = pd.DataFrame(upload_rows, columns=FD_SLOT_ORDER)
    portfolio_summary_df = pd.DataFrame(summary_rows)

    n_sel = max(len(selected), 1)
    exp_rows = []
    for i in range(n_players):
        if player_counts[i] > 0:
            exp_pct = (player_counts[i] / n_sel) * 100.0
            p_own = float(own[i])
            exp_rows.append(
                {
                    "fd_id": fd_ids[i],
                    "name": names[i],
                    "team": teams[i],
                    "pos": slate_df["pos_raw"].iloc[i],
                    "salary": int(slate_df["salary"].iloc[i]),
                    "proj_mu": round(float(slate_df["proj_mu"].iloc[i]), 2),
                    "proj_p99": round(float(slate_df["proj_p99"].iloc[i]), 2),
                    "proj_own_pct": round(p_own, 1),
                    "portfolio_exp_pct": round(exp_pct, 1),
                    "leverage_pct": round(exp_pct - p_own, 1),
                    "lineups_count": int(player_counts[i]),
                }
            )
    exposure_df = pd.DataFrame(exp_rows).sort_values("portfolio_exp_pct", ascending=False).reset_index(drop=True)
    return fanduel_upload_df, portfolio_summary_df, exposure_df
