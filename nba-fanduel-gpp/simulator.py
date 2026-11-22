"""Correlated NBA slate Monte Carlo simulator with right-skewed Gamma FanDuel marginals."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

from config import GPPConfig


def _nearest_psd_correlation(corr: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Project a symmetric matrix to the nearest valid positive-definite correlation matrix."""
    sym = 0.5 * (corr + corr.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, eps)
    psd = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(psd), eps, None))
    psd = psd / np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd


def build_nba_correlation_matrix(
    slate_df: pd.DataFrame,
    cfg: Optional[GPPConfig] = None,
    historical_logs: Optional[pd.DataFrame] = None,
    empirical_shrinkage: float = 0.35,
) -> np.ndarray:
    """Construct an N x N NBA player-to-player correlation matrix for the slate.

    Captures structural NBA DFS correlations:
      - Same-game shootout / pace / OT boost (higher in close, high-total games)
      - Teammate PG <-> Big/Wing assist synergy (+)
      - Teammate competing high-usage scorers cannibalization (-)
      - Starter <-> same-position backup minutes/foul-trouble swap (-)
      - Opponent primary scorer bring-back in close games (+)
    Optionally blends with empirical historical residual covariance when available.
    """
    cfg = cfg or GPPConfig()
    n = len(slate_df)
    corr = np.eye(n, dtype=float)
    if n <= 1:
        return corr

    teams = slate_df["team"].to_numpy()
    games = slate_df["game"].to_numpy()
    pos = slate_df["primary_pos"].to_numpy()
    sal = slate_df["salary"].to_numpy(dtype=float)
    proj_min = slate_df["proj_min"].to_numpy(dtype=float)
    spread_abs = slate_df.get("spread_abs", pd.Series(5.0, index=slate_df.index)).to_numpy(dtype=float)
    vegas_tot = slate_df.get("vegas_total", pd.Series(226.0, index=slate_df.index)).to_numpy(dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if games[i] != games[j]:
                continue

            # Close, high-total games have stronger shared pace / OT correlation
            closeness = float(np.clip(1.0 - spread_abs[i] / 16.0, 0.25, 1.0))
            tot_scale = float(np.clip((vegas_tot[i] - 210.0) / 30.0, 0.5, 1.25))
            base_game_rho = cfg.rho_same_game_shootout * closeness * tot_scale

            if teams[i] == teams[j]:
                rho = base_game_rho
                # Starter <-> same-position backup minutes swap (foul trouble / blowout)
                if pos[i] == pos[j] and (
                    (proj_min[i] >= 28 and proj_min[j] <= 24)
                    or (proj_min[j] >= 28 and proj_min[i] <= 24)
                ):
                    rho += cfg.rho_starter_backup_swap
                # PG <-> C/PF/SG assist-finisher synergy
                elif ("PG" in (pos[i], pos[j])) and any(p in (pos[i], pos[j]) for p in ("C", "PF", "SF")):
                    rho += cfg.rho_teammate_assist_synergy
                # Two high-salary scorers on same team cannibalize shot volume slightly
                elif sal[i] >= 7500 and sal[j] >= 7500:
                    rho += cfg.rho_teammate_usage_cannibal
            else:
                # Opponent bring-back: opposing stars push each other in close shootouts
                rho = base_game_rho * 0.65
                if sal[i] >= 7000 and sal[j] >= 7000 and spread_abs[i] <= 6.5:
                    rho += cfg.rho_opp_bringback_close * closeness

            rho = float(np.clip(rho, -0.55, 0.55))
            corr[i, j] = rho
            corr[j, i] = rho

    # Optional empirical residual correlation blending (shrunk toward structured prior)
    if historical_logs is not None and not historical_logs.empty and "eid" in historical_logs.columns:
        piv = historical_logs.pivot_table(index="eid", columns="player_key", values="fd_pts")
        keys = slate_df["player_key"].tolist()
        common = [k for k in keys if k in piv.columns]
        if len(common) >= 4 and len(piv) >= 15:
            emp = piv[common].corr(min_periods=8).to_numpy()
            key_to_idx = {k: idx for idx, k in enumerate(keys)}
            for a_i, ka in enumerate(common):
                for b_i in range(a_i + 1, len(common)):
                    kb = common[b_i]
                    val = emp[a_i, b_i]
                    if np.isfinite(val):
                        ii, jj = key_to_idx[ka], key_to_idx[kb]
                        blended = (1.0 - empirical_shrinkage) * corr[ii, jj] + empirical_shrinkage * float(val)
                        corr[ii, jj] = corr[jj, ii] = float(np.clip(blended, -0.55, 0.55))

    return _nearest_psd_correlation(corr)


def simulate_slate_scores(
    slate_df: pd.DataFrame,
    cfg: Optional[GPPConfig] = None,
    historical_logs: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate (n_sims x n_players) correlated FanDuel NBA scores using a Gaussian copula
    with Gamma right-skewed marginals.

    Returns (sim_matrix, corr_matrix).
    """
    cfg = cfg or GPPConfig()
    rng = np.random.default_rng(cfg.random_seed)
    n_players = len(slate_df)
    n_sims = cfg.n_sims

    corr = build_nba_correlation_matrix(slate_df, cfg=cfg, historical_logs=historical_logs)
    L = np.linalg.cholesky(corr)

    # Correlated standard normal draws -> uniform quantiles via Gaussian copula
    z = rng.standard_normal(size=(n_sims, n_players)) @ L.T
    u = np.clip(norm.cdf(z), 1e-5, 1.0 - 1e-5)

    mu = slate_df["proj_mu"].to_numpy(dtype=float)
    sd = slate_df["proj_sd"].to_numpy(dtype=float)
    out_prob = slate_df.get("out_prob", pd.Series(0.0, index=slate_df.index)).to_numpy(dtype=float)

    sim_matrix = np.zeros((n_sims, n_players), dtype=float)
    for j in range(n_players):
        if mu[j] <= 0.1 or sd[j] <= 0.1 or out_prob[j] >= 0.90:
            sim_matrix[:, j] = 0.0
            continue
        # Gamma distribution matching mean mu[j] and std sd[j]:
        # k = (mu/sd)^2, theta = sd^2 / mu  -> naturally non-negative & right-skewed!
        k_shape = max((mu[j] / sd[j]) ** 2, 1.5)
        theta_scale = (sd[j] ** 2) / mu[j]
        draws = gamma.ppf(u[:, j], a=k_shape, scale=theta_scale)
        # Apply Questionable/Doubtful scratch risk independently per sim
        if 0.05 < out_prob[j] < 0.90:
            active_draw = rng.random(n_sims) >= out_prob[j]
            draws = np.where(active_draw, draws, 0.0)
        sim_matrix[:, j] = draws

    return sim_matrix, corr
