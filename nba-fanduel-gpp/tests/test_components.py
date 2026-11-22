"""Component tests: prop de-vig, injury report parsing, ESPN ingestion, FanDuel CSV loading,
no look-ahead in the walk-forward, team minutes cap, and hard portfolio limits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import GPPConfig
from data_sources import (
    build_espn_fanduel_logs,
    compute_prop_implied_fd_prior,
    load_fanduel_slate,
    parse_injury_report_text,
)
from optimizer import build_gpp_portfolio
from projections import build_slate_projections, project_players, walk_forward_backtest
from run_pipeline import _build_demo_datasets
from simulator import simulate_slate_scores


def _espn_summary(home, away, home_spread, box):
    keys = ["minutes", "points", "fieldGoalsMade-fieldGoalsAttempted", "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
            "freeThrowsMade-freeThrowsAttempted", "rebounds", "assists", "turnovers", "steals", "blocks",
            "offensiveRebounds", "defensiveRebounds", "fouls", "plusMinus"]
    players = []
    for team, rows in box.items():
        players.append({"team": {"abbreviation": team}, "statistics": [{"keys": keys, "athletes": [
            {"athlete": {"id": str(i), "displayName": n, "position": {"abbreviation": "G"}}, "starter": True, "stats": s}
            for i, (n, s) in enumerate(rows)]}]})
    return {"header": {"competitions": [{"competitors": [
        {"homeAway": "home", "team": {"abbreviation": home}}, {"homeAway": "away", "team": {"abbreviation": away}}]}]},
        "pickcenter": [{"overUnder": 230.5, "spread": home_spread}], "boxscore": {"players": players}}


class TestComponents(unittest.TestCase):
    def test_prop_devig(self) -> None:
        props = pd.DataFrame([
            {"player": "Shai Gilgeous-Alexander", "stat": "pts", "line": 31.5, "over_odds": -110, "under_odds": -110},
            {"player": "Shai Gilgeous-Alexander", "stat": "ast", "line": 6.5, "over_odds": -150, "under_odds": 120},
        ])
        out = compute_prop_implied_fd_prior(props).set_index("player_key")
        row = out.loc["shai gilgeous alexander"]
        self.assertAlmostEqual(row["mkt_pts"], 31.5, places=6)  # fair price -> mean at the line
        self.assertGreater(row["mkt_ast"], 6.5)                   # over favoured -> mean above the line
        self.assertEqual(int(row["n_markets"]), 2)
        self.assertAlmostEqual(row["vegas_fd_prior"], row["mkt_pts"] + 1.5 * row["mkt_ast"], places=6)

    def test_injury_report_text(self) -> None:
        text = ("11/22/2026 07:00 (ET) POR@SAC Portland Trail Blazers Morant, Ja Out Injury/Illness - Left Hamstring; Strain\n"
                "Sharpe, Shaedon Questionable Injury/Illness - Knee\n"
                "Sacramento Kings Sabonis, Domantas Probable Injury/Illness - Left Knee; Soreness\n"
                "Acuff Jr., Darius Available\n")
        df = parse_injury_report_text(text).set_index("player_key")
        self.assertEqual(df.loc["ja morant", "status"], "OUT")
        self.assertEqual(df.loc["ja morant", "team"], "POR")
        self.assertEqual(df.loc["shaedon sharpe", "team"], "POR")
        self.assertAlmostEqual(df.loc["shaedon sharpe", "out_prob"], 0.35)
        self.assertEqual(df.loc["domantas sabonis", "team"], "SAC")
        self.assertEqual(df.loc["darius acuff", "out_prob"], 0.0)

    def test_espn_logs_flags_and_spread_sign(self) -> None:
        line = lambda pts, stl, blk: ["30", str(pts), "8-15", "2-5", "4-4", "5", "3", "2", str(stl), str(blk), "1", "4", "2", "+5"]
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "events.json").write_text(json.dumps({
                "1": {"date": "20261120", "status": "STATUS_FINAL"},
                "2": {"date": "20261121", "status": "STATUS_FINAL"},
                "3": {"date": "20261122", "status": "STATUS_SCHEDULED"}}))
            (d / "sum_1.json").write_text(json.dumps(_espn_summary("SA", "GS", -4.5, {"SA": [("Victor Wembanyama", line(22, 1, 5))], "GS": [("Stephen Curry", line(30, 2, 0))]})))
            (d / "sum_2.json").write_text(json.dumps(_espn_summary("GS", "UTAH", 2.0, {"GS": [("Stephen Curry", line(25, 1, 1))], "UTAH": [("Lauri Markkanen", line(20, 0, 1))]})))
            (d / "sum_3.json").write_text(json.dumps({**_espn_summary("DEN", "GS", -6.0, {}), "boxscore": {}}))
            logs, games = build_espn_fanduel_logs(d)
        curry = logs[logs["player_key"] == "stephen curry"].set_index("date")
        self.assertEqual(set(logs["team"]), {"SAS", "GSW", "UTA"})
        self.assertAlmostEqual(curry.loc["2026-11-20", "spread"], 4.5)   # away underdog: +4.5
        self.assertAlmostEqual(curry.loc["2026-11-21", "spread"], 2.0)   # home: home spread as-is
        self.assertEqual(int(curry.loc["2026-11-21", "is_b2b"]), 1)
        # 30 + 5*1.2 + 3*1.5 + 2*3 + 0*3 - 2 = 44.5
        self.assertAlmostEqual(curry.loc["2026-11-20", "fd_pts"], 44.5)
        self.assertEqual(len(games), 3)                                    # scheduled game kept for Vegas lines

    def test_fanduel_csv_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fd.csv"
            pd.DataFrame([
                {"Id": "123-1", "Position": "PG/SG", "First Name": "Stephen", "Nickname": "Stephen Curry", "Last Name": "Curry",
                 "FPPG": 44.1, "Played": 15, "Salary": 9800, "Game": "GS@DEN", "Team": "GS", "Opponent": "DEN", "Injury Indicator": "Q"},
                {"Id": "123-2", "Position": "C", "First Name": "Nikola", "Nickname": "Nikola Jokić", "Last Name": "Jokić",
                 "FPPG": 60.2, "Played": 15, "Salary": 12500, "Game": "GS@DEN", "Team": "DEN", "Opponent": "GS", "Injury Indicator": None},
            ]).to_csv(p, index=False)
            df = load_fanduel_slate(p).set_index("player_key")
        self.assertEqual(df.loc["stephen curry", "positions"], ("PG", "SG"))
        self.assertEqual(df.loc["stephen curry", "game"], "GSW@DEN")
        self.assertEqual(df.loc["stephen curry", "fd_injury"], "Q")
        self.assertEqual(df.loc["nikola jokic", "team"], "DEN")

    def test_walk_forward_has_no_lookahead(self) -> None:
        _, logs, db_teams, db_players, *_ = _build_demo_datasets(seed=3)
        _, oos_a = walk_forward_backtest(logs, db_teams, db_players, min_train_dates=5, return_oos=True)
        last = logs["date"].max()
        tampered = logs.copy()
        tampered.loc[tampered["date"] == last, ["pts", "fd_pts"]] *= 10
        _, oos_b = walk_forward_backtest(tampered, db_teams, db_players, min_train_dates=5, return_oos=True)
        early_a = oos_a[oos_a["date"] < last].sort_values(["date", "player_key"])["fd_struct"].to_numpy()
        early_b = oos_b[oos_b["date"] < last].sort_values(["date", "player_key"])["fd_struct"].to_numpy()
        np.testing.assert_allclose(early_a, early_b)

    def test_team_minutes_cap(self) -> None:
        rows = []
        for day in range(1, 6):
            for i in range(12):  # 12 players averaging 30 minutes = 360 minutes a night
                rows.append({"eid": f"g{day}", "date": f"2026-11-0{day}", "player_key": f"p{i}", "team": "AAA", "opp": "BBB",
                             "pos": "G", "min": 30.0, "pts": 15, "reb": 5, "ast": 3, "stl": 1, "blk": 0, "tov": 2})
        hist = pd.DataFrame(rows)
        targets = pd.DataFrame({"player_key": [f"p{i}" for i in range(12)], "team": "AAA", "opp": "BBB",
                                "date": "2026-11-06", "out_prob": 0.0})
        proj = project_players(hist, targets, pd.DataFrame(), GPPConfig())
        self.assertAlmostEqual(float(proj["proj_min"].sum()), 240.0, places=3)

    def test_portfolio_limits_hold_with_tiny_candidate_pool(self) -> None:
        slate, logs, db_teams, db_players, db_wowy, vegas_games, vegas_props = _build_demo_datasets(seed=11)
        cfg = GPPConfig(n_sims=200, n_lineups=8, candidate_pool_size=3, max_exposure=0.5, min_unique_players=3, random_seed=11)
        proj = build_slate_projections(slate, logs, db_teams, db_players, db_wowy, None, vegas_games, vegas_props, None, cfg)
        proj["proj_own"] = 8.0
        sims, _ = simulate_slate_scores(proj, cfg=cfg)
        upload, _, exposure = build_gpp_portfolio(proj, sims, cfg=cfg)
        self.assertEqual(len(upload), 8)
        self.assertLessEqual(int(exposure["lineups_count"].max()), 4)
        sets = [frozenset(str(v).split(":")[0] for v in r) for r in upload.itertuples(index=False)]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertGreaterEqual(len(sets[i] - sets[j]), 3)


if __name__ == "__main__":
    unittest.main()
