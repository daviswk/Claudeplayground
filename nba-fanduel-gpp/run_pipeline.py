"""End-to-end CLI entrypoint for the FanDuel NBA Classic GPP Lineup Builder."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import FD_SCORING, GPPConfig
from data_sources import (
    build_espn_fanduel_logs,
    derive_team_factors_from_logs,
    compute_fd_points,
    compute_prop_implied_fd_prior,
    estimate_slate_ownership,
    fetch_databallr_teams,
    fetch_nba_injury_report,
    fetch_odds_api_nba,
    load_databallr_players,
    load_databallr_wowy,
    load_fanduel_slate,
    parse_nba_injury_pdf,
)
from normalize import normalize_player_name, parse_fd_positions
from optimizer import build_gpp_portfolio
from projections import build_slate_projections, fit_ridge_on_oos, walk_forward_backtest
from simulator import simulate_slate_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_demo_datasets(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic 6-game FanDuel Classic demo slate (fake players, salaries, props
    and box scores) for tests and smoke runs. Team ratings are illustrative inputs, not a
    verified Databallr snapshot.
    """
    rng = np.random.default_rng(seed)

    # Illustrative six-factor team ratings in Databallr's column layout (not verified against databallr.com)
    db_teams = pd.DataFrame(
        [
            {"team": "SAS", "ortg": 118.2, "drtg": 107.3, "net": 11.0, "ots": -1.6, "otov": 3.0, "orb": 2.8, "dts": 6.6, "dtov": -1.4, "drb": 1.6, "three_p_luck": 1.2, "stop": 2.1, "pace": 100.4, "def_rim_freq": 30.5, "def_ftr": 0.22},
            {"team": "BOS", "ortg": 117.7, "drtg": 107.3, "net": 10.4, "ots": 1.7, "otov": 1.1, "orb": 0.8, "dts": 5.4, "dtov": 1.4, "drb": -0.1, "three_p_luck": 0.8, "stop": 1.9, "pace": 98.2, "def_rim_freq": 31.2, "def_ftr": 0.21},
            {"team": "MEM", "ortg": 121.6, "drtg": 112.6, "net": 9.0, "ots": 7.3, "otov": -0.8, "orb": 1.0, "dts": 1.9, "dtov": -1.6, "drb": 1.2, "three_p_luck": -0.5, "stop": 1.5, "pace": 102.8, "def_rim_freq": 34.0, "def_ftr": 0.25},
            {"team": "LAL", "ortg": 120.2, "drtg": 113.6, "net": 6.6, "ots": 6.8, "otov": -1.2, "orb": 0.6, "dts": -1.4, "dtov": 3.3, "drb": -1.4, "three_p_luck": -1.1, "stop": 0.8, "pace": 101.2, "def_rim_freq": 34.8, "def_ftr": 0.24},
            {"team": "OKC", "ortg": 113.3, "drtg": 109.5, "net": 3.9, "ots": -2.4, "otov": 1.9, "orb": -0.2, "dts": 6.4, "dtov": -0.2, "drb": -1.6, "three_p_luck": 0.4, "stop": 3.2, "pace": 100.9, "def_rim_freq": 30.0, "def_ftr": 0.23},
            {"team": "DEN", "ortg": 115.6, "drtg": 114.3, "net": 1.2, "ots": 0.9, "otov": 0.4, "orb": 0.2, "dts": 2.6, "dtov": -1.0, "drb": -1.9, "three_p_luck": -0.9, "stop": -0.4, "pace": 99.8, "def_rim_freq": 35.5, "def_ftr": 0.24},
            {"team": "POR", "ortg": 116.6, "drtg": 111.1, "net": 5.5, "ots": 4.5, "otov": -0.5, "orb": -1.5, "dts": 2.3, "dtov": -1.4, "drb": 2.1, "three_p_luck": 0.3, "stop": 0.9, "pace": 101.0, "def_rim_freq": 33.4, "def_ftr": 0.25},
            {"team": "SAC", "ortg": 107.6, "drtg": 121.3, "net": -13.7, "ots": -2.6, "otov": -1.8, "orb": -2.1, "dts": -6.5, "dtov": -0.4, "drb": -0.3, "three_p_luck": -2.4, "stop": -2.8, "pace": 101.6, "def_rim_freq": 36.8, "def_ftr": 0.27},
            {"team": "CHI", "ortg": 111.7, "drtg": 120.0, "net": -8.3, "ots": -0.7, "otov": -1.7, "orb": 0.0, "dts": -2.8, "dtov": -0.7, "drb": -2.3, "three_p_luck": -1.8, "stop": -2.1, "pace": 103.1, "def_rim_freq": 37.2, "def_ftr": 0.26},
            {"team": "WAS", "ortg": 110.5, "drtg": 117.9, "net": -7.5, "ots": -5.3, "otov": 1.0, "orb": 0.7, "dts": -0.8, "dtov": -2.3, "drb": -0.7, "three_p_luck": -1.5, "stop": -1.9, "pace": 102.4, "def_rim_freq": 36.0, "def_ftr": 0.27},
            {"team": "PHI", "ortg": 114.9, "drtg": 112.7, "net": 2.2, "ots": 0.1, "otov": 2.5, "orb": -1.7, "dts": 1.9, "dtov": -0.4, "drb": -0.2, "three_p_luck": 0.2, "stop": 0.7, "pace": 99.1, "def_rim_freq": 32.8, "def_ftr": 0.24},
            {"team": "MIL", "ortg": 107.4, "drtg": 120.1, "net": -12.6, "ots": -5.5, "otov": -1.8, "orb": 0.7, "dts": -2.2, "dtov": -2.6, "drb": -1.2, "three_p_luck": -1.3, "stop": -1.8, "pace": 100.5, "def_rim_freq": 35.1, "def_ftr": 0.25},
        ]
    )

    vegas_games = pd.DataFrame(
        [
            {"home_team": "MEM", "away_team": "CHI", "total": 241.5, "home_spread": -7.5},
            {"home_team": "LAL", "away_team": "DEN", "total": 234.5, "home_spread": -2.5},
            {"home_team": "SAC", "away_team": "POR", "total": 231.0, "home_spread": 5.5},
            {"home_team": "PHI", "away_team": "MIL", "total": 227.5, "home_spread": -6.0},
            {"home_team": "SAS", "away_team": "WAS", "total": 229.0, "home_spread": -11.5},
            {"home_team": "BOS", "away_team": "OKC", "total": 222.5, "home_spread": -2.0},
        ]
    )

    # 8-player rotation per team across 12 teams (96 players)
    roster_templates = [
        ("PG", "PG/SG", 34.5, 9200, 23.5, 4.8, 7.6, 1.4, 0.4, 3.1, 3.4),
        ("SG", "SG/SF", 33.0, 7800, 20.5, 4.2, 4.0, 1.2, 0.5, 2.2, 2.2),
        ("SF", "SF/PF", 32.0, 6900, 16.5, 5.8, 3.1, 1.3, 0.8, 1.8, 1.8),
        ("PF", "PF/C",  31.5, 7200, 17.0, 8.2, 2.8, 0.9, 1.2, 1.9, 1.5),
        ("C",  "C",     32.5, 8600, 18.5, 11.4, 3.6, 0.8, 2.1, 2.3, 2.9),
        ("PG", "PG",    23.0, 5100, 11.2, 2.8, 4.6, 0.9, 0.2, 1.5, 0.4),
        ("SG", "SG/SF", 22.0, 4600, 10.5, 3.2, 2.0, 0.8, 0.4, 1.1, 0.1),
        ("PF", "PF/C",  20.5, 4200, 8.8,  5.6, 1.3, 0.6, 1.0, 1.0, -0.2),
    ]

    slate_rows, db_player_rows, wowy_rows, prop_rows = [], [], [], []
    pid_counter = 1000
    for _, g in vegas_games.iterrows():
        for tm, opp in [(g["home_team"], g["away_team"]), (g["away_team"], g["home_team"])]:
            tm_net = float(db_teams.loc[db_teams["team"] == tm, "net"].iloc[0])
            star_boost = 1.0 + np.clip(tm_net / 80.0, -0.10, 0.12)
            for slot_idx, (ppos, pos_raw, avg_m, base_sal, pts, reb, ast, stl, blk, tov, base_dpm) in enumerate(roster_templates):
                pid_counter += 1
                pname = f"{tm} {ppos}{slot_idx + 1}"
                # Inject realistic stars & injury value
                inj_tag = "O" if (tm == "MEM" and slot_idx == 0) or (tm == "PHI" and slot_idx == 4) else ""
                sal = int(round((base_sal * rng.uniform(0.92, 1.08)) / 100.0) * 100)
                fppg = (pts * 1.0 + reb * 1.2 + ast * 1.5 + stl * 3.0 + blk * 3.0 - tov * 1.0) * star_boost
                slate_rows.append(
                    {
                        "fd_id": f"FD-{pid_counter}",
                        "name": pname,
                        "player_key": normalize_player_name(pname),
                        "pos_raw": pos_raw,
                        "positions": parse_fd_positions(pos_raw),
                        "salary": max(3500, sal),
                        "fppg": round(fppg, 1),
                        "team": tm,
                        "opp": opp,
                        "game": f"{g['away_team']}@{g['home_team']}",
                        "fd_injury": inj_tag,
                        "primary_pos": ppos,
                    }
                )
                db_player_rows.append(
                    {
                        "player_key": normalize_player_name(pname),
                        "team": tm,
                        "dpm": round(base_dpm + rng.normal(0, 0.6), 2),
                        "o_dpm": round(base_dpm * 0.65, 2),
                        "d_dpm": round(base_dpm * 0.35, 2),
                        "zts": round(rng.normal(0.5, 2.0), 2),
                        "rapm": round(base_dpm * 0.9, 2),
                        "usg_db": round(18.0 + (10.0 if slot_idx < 2 else 3.0), 1),
                    }
                )
                # Add Databallr WOWY shifts for teammates of OUT stars
                if tm == "MEM" and slot_idx in (1, 5):
                    wowy_rows.append(
                        {
                            "player_key": normalize_player_name(pname),
                            "teammate_out_key": normalize_player_name("MEM PG1"),
                            "team": "MEM",
                            "off_min": 310.0,
                            "shrink_weight": 310.0 / (310.0 + 266.0),
                            "fd_p36_shift_shrunk": 5.4 if slot_idx == 5 else 4.2,
                        }
                    )
                if tm == "PHI" and slot_idx in (3, 7):
                    wowy_rows.append(
                        {
                            "player_key": normalize_player_name(pname),
                            "teammate_out_key": normalize_player_name("PHI C5"),
                            "team": "PHI",
                            "off_min": 280.0,
                            "shrink_weight": 280.0 / (280.0 + 266.0),
                            "fd_p36_shift_shrunk": 6.1 if slot_idx == 7 else 3.8,
                        }
                    )
                if slot_idx < 5 and not inj_tag:
                    for st, val in [("pts", pts), ("reb", reb), ("ast", ast), ("stl", stl), ("blk", blk)]:
                        prop_rows.append(
                            {
                                "player": pname,
                                "stat": st,
                                "line": round(val * star_boost * rng.uniform(0.95, 1.05), 1),
                                "over_odds": -110.0,
                                "under_odds": -110.0,
                            }
                        )

    slate_df = pd.DataFrame(slate_rows)
    db_players_df = pd.DataFrame(db_player_rows)
    db_wowy_df = pd.DataFrame(wowy_rows)
    vegas_props_prior = compute_prop_implied_fd_prior(pd.DataFrame(prop_rows))

    # Build 12 historical slates of boxscore logs for walk-forward CV & empirical rates
    log_rows = []
    opp_dts_map = db_teams.set_index("team")["dts"].to_dict()
    for day in range(1, 13):
        date_str = f"2026-11-{day:02d}"
        for _, p in slate_df.iterrows():
            slot_idx = int(p["name"][-1]) - 1
            _, _, avg_m, _, pts, reb, ast, stl, blk, tov, _ = roster_templates[slot_idx]
            opp_edge = -0.015 * opp_dts_map.get(p["opp"], 0.0)
            m_draw = float(np.clip(rng.normal(avg_m, 3.8), 10.0, 42.0))
            scale = (m_draw / avg_m) * (1.0 + opp_edge)
            row = {
                "eid": f"E{day}_{p['game']}",
                "date": date_str,
                "player_key": p["player_key"],
                "name": p["name"],
                "team": p["team"],
                "opp": p["opp"],
                "home": 1 if p["game"].endswith(p["team"]) else 0,
                "pos": p["primary_pos"],
                "starter": 1 if slot_idx < 5 else 0,
                "min": m_draw,
                "pts": max(0.0, float(rng.normal(pts * scale, max(pts * 0.28, 2.0)))),
                "reb": max(0.0, float(rng.normal(reb * scale, max(reb * 0.30, 1.2)))),
                "ast": max(0.0, float(rng.normal(ast * scale, max(ast * 0.32, 1.0)))),
                "stl": float(rng.poisson(max(stl * scale, 0.1))),
                "blk": float(rng.poisson(max(blk * scale, 0.1))),
                "tov": float(rng.poisson(max(tov * scale, 0.2))),
                "total": 228.0 + float(rng.normal(0, 6.0)),
                "spread": float(rng.normal(0, 5.5)),
                "is_b2b": int(day % 4 == 0),
            }
            log_rows.append(row)

    logs_df = pd.DataFrame(log_rows)
    logs_df["fd_pts"] = compute_fd_points(logs_df)
    return slate_df, logs_df, db_teams, db_players_df, db_wowy_df, vegas_games, vegas_props_prior


def run_pipeline(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = GPPConfig(
        n_sims=args.n_sims,
        n_lineups=args.n_lineups,
        max_exposure=args.max_exposure,
        min_unique_players=args.min_unique,
    )

    if args.demo:
        logger.info("Running in --demo mode on a synthetic 6-game slate.")
        slate_df, player_logs, db_teams, db_players, db_wowy, vegas_games, vegas_props_prior = _build_demo_datasets()
        injuries_df = pd.DataFrame()
    else:
        if not args.fanduel_csv:
            raise ValueError("--fanduel-csv is required unless --demo is specified.")
        slate_df = load_fanduel_slate(args.fanduel_csv)
        player_logs, espn_games = build_espn_fanduel_logs(args.espn_dir) if args.espn_dir else (pd.DataFrame(), pd.DataFrame())
        if args.slate_date and not player_logs.empty:
            player_logs = player_logs[player_logs["date"] < args.slate_date].reset_index(drop=True)
        db_teams = fetch_databallr_teams(csv_path=args.databallr_teams_csv, live=args.live_fetch)
        if db_teams.empty and not player_logs.empty:
            logger.info("No Databallr team table; deriving six-factor team stats from ESPN box scores.")
            db_teams = derive_team_factors_from_logs(player_logs)
        db_players = load_databallr_players(args.databallr_players_csv)
        db_wowy = load_databallr_wowy(args.databallr_wowy_csv)

        if args.injury_pdf:
            injuries_df = parse_nba_injury_pdf(args.injury_pdf)
        elif args.live_fetch and args.slate_date:
            injuries_df = fetch_nba_injury_report(args.slate_date, out_dir / "injuries")
        else:
            injuries_df = pd.DataFrame()

        if args.live_fetch:
            vegas_games, props_raw = fetch_odds_api_nba(
                max_credits=cfg.odds_api_max_credits_per_run,
                min_remaining=cfg.odds_api_min_remaining_credits,
            )
            vegas_props_prior = compute_prop_implied_fd_prior(props_raw)
        else:
            vegas_games = pd.read_csv(args.vegas_games_csv) if args.vegas_games_csv else pd.DataFrame()
            vegas_props_prior = (
                compute_prop_implied_fd_prior(pd.read_csv(args.vegas_props_csv))
                if args.vegas_props_csv
                else pd.DataFrame()
            )
        if vegas_games.empty and not espn_games.empty and args.slate_date:
            vegas_games = espn_games[espn_games["date"] == args.slate_date]
            logger.info("Using ESPN pick-center lines for %d games.", len(vegas_games))

    # 1. Strict expanding-window walk-forward CV on historical logs (also yields out-of-sample residuals)
    wf_metrics, oos = walk_forward_backtest(player_logs, db_teams, db_players, cfg=cfg, return_oos=True) \
        if not player_logs.empty else ({}, pd.DataFrame())
    (out_dir / "walk_forward_cv.json").write_text(json.dumps(wf_metrics, indent=2), encoding="utf-8")
    logger.info("Walk-Forward CV Metrics: %s", {k: v for k, v in wf_metrics.items() if not isinstance(v, dict)})

    # 2. Ridge correction fit on the structural model's out-of-sample residuals (no double counting
    #    of matchup effects the structural model already prices); projection spread from the same residuals
    ridge_model = fit_ridge_on_oos(oos, cfg) if not oos.empty else None
    sd_curve = tuple(wf_metrics["sd_curve"]) if wf_metrics.get("sd_curve") else None

    # 3. Project tonight's FanDuel slate + ownership
    proj_df = build_slate_projections(
        slate_df=slate_df,
        player_logs=player_logs,
        databallr_teams=db_teams,
        databallr_players=db_players,
        databallr_wowy=db_wowy,
        injuries_df=injuries_df,
        vegas_games_df=vegas_games,
        vegas_props_prior=vegas_props_prior,
        ridge_model=ridge_model,
        cfg=cfg,
        slate_date=args.slate_date or None,
        sd_curve=sd_curve,
    )
    proj_df["proj_own"] = estimate_slate_ownership(proj_df, ext_ownership_csv=args.ownership_csv, min_join_rate=cfg.min_join_rate)
    proj_df.to_csv(out_dir / "slate_projections.csv", index=False)

    # 4. Run correlated Monte Carlo slate simulation
    sim_matrix, _ = simulate_slate_scores(proj_df, cfg=cfg, historical_logs=player_logs)

    # 5. Build diversified FanDuel Classic GPP portfolio via HiGHS MILP
    upload_df, summary_df, exposure_df = build_gpp_portfolio(proj_df, sim_matrix, cfg=cfg)
    upload_df.to_csv(out_dir / "fanduel_upload.csv", index=False)
    summary_df.to_csv(out_dir / "portfolio_summary.csv", index=False)
    exposure_df.to_csv(out_dir / "exposure_report.csv", index=False)

    logger.info("Generated %d FanDuel Classic GPP lineups in %s", len(upload_df), out_dir)
    logger.info("Top 5 Portfolio Lineups:\n%s", summary_df[["lineup_id", "salary", "mean_pts", "p99_pts", "top_1pct_rate", "gpp_ev", "sum_own_pct", "primary_game_stack"]].head(5).to_string(index=False))
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FanDuel NBA Classic GPP Portfolio Builder (Databallr + Monte Carlo + HiGHS)")
    parser.add_argument("--demo", action="store_true", help="Run self-contained 6-game 2026-27 NBA demo slate")
    parser.add_argument("--live-fetch", action="store_true", help="Fetch live Databallr /teams, NBA Injury PDF, and The Odds API")
    parser.add_argument("--slate-date", type=str, default="", help="Slate date YYYY-MM-DD (for NBA Official Injury Report)")
    parser.add_argument("--fanduel-csv", type=str, default="", help="Path to official FanDuel player list CSV")
    parser.add_argument("--espn-dir", type=str, default="", help="Path to nba-props ESPN JSON directory (events.json + sum_*.json)")
    parser.add_argument("--databallr-teams-csv", type=str, default="", help="Path to Databallr Team Six-Factor CSV")
    parser.add_argument("--databallr-players-csv", type=str, default="", help="Path to Databallr Player Stats/DPM CSV")
    parser.add_argument("--databallr-wowy-csv", type=str, default="", help="Path to Databallr WOWY / Stat Line Shift CSV")
    parser.add_argument("--injury-pdf", type=str, default="", help="Path to local NBA Official Injury Report PDF")
    parser.add_argument("--vegas-games-csv", type=str, default="", help="Path to Vegas game totals/spreads CSV")
    parser.add_argument("--vegas-props-csv", type=str, default="", help="Path to Vegas player props CSV")
    parser.add_argument("--ownership-csv", type=str, default="", help="Optional external ownership projections CSV")
    parser.add_argument("--output-dir", type=str, default="results_gpp", help="Output directory for FanDuel CSV & reports")
    parser.add_argument("--n-sims", type=int, default=2000, help="Number of correlated Monte Carlo slate simulations")
    parser.add_argument("--n-lineups", type=int, default=20, help="Number of GPP lineups to generate")
    parser.add_argument("--max-exposure", type=float, default=0.45, help="Max player exposure across portfolio (0.0-1.0)")
    parser.add_argument("--min-unique", type=int, default=2, help="Minimum unique players between any two lineups")
    return parser


def main() -> None:
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
