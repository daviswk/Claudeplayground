"""Automated verification suite for the FanDuel NBA Classic GPP pipeline."""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from config import FD_SALARY_CAP, FD_SLOT_ORDER, GPPConfig
from data_sources import compute_fd_points, load_databallr_wowy
from normalize import audit_join_rate, normalize_player_name, normalize_team, parse_fd_positions
from optimizer import build_gpp_portfolio, solve_single_fanduel_lineup
from projections import build_slate_projections, walk_forward_backtest
from run_pipeline import _build_demo_datasets, run_pipeline
from simulator import simulate_slate_scores


class TestFanDuelNBAGPP(unittest.TestCase):
    def test_normalization_and_join_audit(self) -> None:
        self.assertEqual(normalize_player_name("Nikola Jokić"), "nikola jokic")
        self.assertEqual(normalize_player_name("Luka Dončić"), "luka doncic")
        self.assertEqual(normalize_player_name("Jaren Jackson Jr."), "jaren jackson")
        self.assertEqual(normalize_player_name("Doncic, Luka"), "luka doncic")
        self.assertEqual(normalize_team("GS"), "GSW")
        self.assertEqual(normalize_team("WSH"), "WAS")
        self.assertEqual(normalize_team("San Antonio Spurs"), "SAS")
        self.assertEqual(parse_fd_positions("PG/SG"), ("PG", "SG"))

        rate, unmatched = audit_join_rate(["nikola jokic", "luka doncic"], ["nikola jokic", "luka doncic"], "Test", strict=True)
        self.assertEqual(rate, 1.0)
        self.assertEqual(unmatched, [])

    def test_fanduel_scoring_exact(self) -> None:
        df = pd.DataFrame([{"pts": 25, "reb": 10, "ast": 8, "stl": 2, "blk": 3, "tov": 4}])
        # 25*1 + 10*1.2 + 8*1.5 + 2*3 + 3*3 - 4*1 = 25 + 12 + 12 + 6 + 9 - 4 = 60.0
        self.assertAlmostEqual(float(compute_fd_points(df).iloc[0]), 60.0, places=6)

    def test_databallr_wowy_bayesian_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wowy.csv"
            pd.DataFrame(
                [
                    {
                        "player": "Desmond Bane",
                        "teammate_out": "Ja Morant",
                        "team": "MEM",
                        "off_min": 266.0,
                        "pts_p36_diff": 6.0,
                        "reb_p36_diff": 0.0,
                        "ast_p36_diff": 2.0,
                        "stl_p36_diff": 0.0,
                        "blk_p36_diff": 0.0,
                        "tov_p36_diff": 1.0,
                    }
                ]
            ).to_csv(p, index=False)
            wowy = load_databallr_wowy(p, prior_minutes=266.0)
            # At 266 min with 266 prior -> exactly 50% kept; raw FD diff = 6 + 3 - 1 = 8 -> shrunk = 4.0
            self.assertAlmostEqual(float(wowy["shrink_weight"].iloc[0]), 0.5, places=6)
            self.assertAlmostEqual(float(wowy["fd_p36_shift_shrunk"].iloc[0]), 4.0, places=6)

    def test_walk_forward_cv_and_end_to_end_portfolio(self) -> None:
        slate_df, logs_df, db_teams, db_players, db_wowy, vegas_games, vegas_props = _build_demo_datasets(seed=7)

        # 1. Walk-forward CV
        wf = walk_forward_backtest(logs_df, db_teams, db_players, min_train_dates=4)
        self.assertGreater(wf["n_eval_rows"], 100)
        self.assertGreater(wf["ridge_spearman"], 0.4)

        # 2. Projections + Simulation + MILP Portfolio
        cfg = GPPConfig(n_sims=400, n_lineups=10, candidate_pool_size=60, max_exposure=0.50, min_unique_players=2, random_seed=7)
        proj = build_slate_projections(slate_df, logs_df, db_teams, db_players, db_wowy, None, vegas_games, vegas_props, None, cfg)
        proj["proj_own"] = 10.0
        sim_mat, corr = simulate_slate_scores(proj, cfg=cfg, historical_logs=logs_df)

        # Verify PSD correlation matrix & non-negative scores
        eigvals = np.linalg.eigvalsh(corr)
        self.assertGreater(float(eigvals.min()), 0.0)
        self.assertGreaterEqual(float(sim_mat.min()), 0.0)

        upload_df, summary_df, exposure_df = build_gpp_portfolio(proj, sim_mat, cfg=cfg)
        self.assertEqual(list(upload_df.columns), FD_SLOT_ORDER)
        self.assertEqual(len(upload_df), 10)

        # Verify FanDuel rules on every generated lineup
        id_to_row = {row["fd_id"]: row for _, row in proj.iterrows()}
        lu_sets = []
        for _, row in upload_df.iterrows():
            ids = [str(v).split(":", 1)[0] for v in row.tolist()]
            self.assertEqual(len(set(ids)), 9)  # 9 distinct players
            sal = sum(int(id_to_row[i]["salary"]) for i in ids)
            self.assertLessEqual(sal, FD_SALARY_CAP)
            teams = pd.Series([id_to_row[i]["team"] for i in ids]).value_counts()
            self.assertLessEqual(int(teams.max()), 4)
            games = {id_to_row[i]["game"] for i in ids}
            self.assertGreaterEqual(len(games), 2)
            # Check slot eligibility
            for slot, pid in zip(FD_SLOT_ORDER, ids):
                self.assertIn(slot, id_to_row[pid]["positions"])
            lu_sets.append(frozenset(ids))

        # Check min_unique between all pairs
        for i in range(len(lu_sets)):
            for j in range(i + 1, len(lu_sets)):
                self.assertGreaterEqual(len(lu_sets[i] - lu_sets[j]), 2)

        # Check max_exposure
        self.assertLessEqual(float(exposure_df["portfolio_exp_pct"].max()), 50.0 + 1e-6)

    def test_cli_demo_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                demo=True,
                live_fetch=False,
                slate_date="",
                fanduel_csv="",
                espn_dir="",
                databallr_teams_csv="",
                databallr_players_csv="",
                databallr_wowy_csv="",
                injury_pdf="",
                vegas_games_csv="",
                vegas_props_csv="",
                ownership_csv="",
                output_dir=tmp,
                n_sims=300,
                n_lineups=6,
                max_exposure=0.50,
                min_unique=2,
            )
            out_dir = run_pipeline(args)
            for fname in ["fanduel_upload.csv", "portfolio_summary.csv", "exposure_report.csv", "slate_projections.csv", "walk_forward_cv.json"]:
                self.assertTrue((out_dir / fname).exists(), f"Missing expected output {fname}")


if __name__ == "__main__":
    unittest.main()
