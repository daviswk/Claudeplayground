# FanDuel NBA Classic GPP Lineup Builder (`nba-fanduel-gpp`)

Extends `../nba-props/` from single-player props to FanDuel NBA Classic tournament (GPP) portfolios:
FanDuel-point projections → correlated Monte Carlo slate simulation → HiGHS MILP lineups → a
diversified portfolio and an upload-ready CSV.

## Quickstart

```bash
cd nba-fanduel-gpp
pip install -r requirements.txt

# Tests (offline, ~20 s)
python3 -m unittest discover -s tests -v

# Synthetic demo slate (fake players and salaries)
python3 run_pipeline.py --demo --n-sims 2000 --n-lineups 20 --output-dir results_demo

# Real slate: FanDuel "Download players list" CSV + ESPN box scores
python3 ../nba-props/fetch_espn.py ../nba-props/data 2026-11-22     # refresh ESPN data through a date
python3 run_pipeline.py \
  --fanduel-csv path/to/FanDuel-NBA-players.csv \
  --espn-dir ../nba-props/data \
  --slate-date 2026-11-22 \
  --n-sims 2000 --n-lineups 20 --max-exposure 0.45 --min-unique 2 \
  --output-dir results_live
```

Optional inputs: `--databallr-teams-csv`, `--databallr-players-csv`, `--databallr-wowy-csv` (Databallr exports),
`--injury-pdf` (official NBA report), `--vegas-games-csv`, `--vegas-props-csv`, `--ownership-csv`.
`--live-fetch` adds the Databallr team page, the official injury PDF and The Odds API (`ODDS_API_KEY`).

## What each step does

| Step | Module | Notes |
|---|---|---|
| Ingest | `data_sources.py` | ESPN summaries → player logs with exact FanDuel points (`PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK − TOV`), team-perspective spread, rest / back-to-back / 3-in-4 / altitude flags. FanDuel CSV loader, Databallr CSV loaders, injury-PDF parser, The Odds API client with a credit circuit breaker, prop de-vig. |
| Team factors | `data_sources.derive_team_factors_from_logs` | Databallr-style ORTG/DRTG/pace/oTS/oTOV/ORB/dTS/dTOV/DRB/3P-luck/STOP computed from ESPN box scores. Used automatically when no Databallr table is supplied. `def_rim_freq` can only come from Databallr. |
| Entity joins | `normalize.py` | Accents, suffixes, `Last, First`, team aliases (`GS`→`GSW`, `UTAH`→`UTA`, full names), FanDuel positions; `audit_join_rate` warns (or raises with `strict=True`) below 95%. |
| Projection | `projections.py` | Per-minute season rates shrunk toward position averages × projected minutes (season + last-5 blend, minutes freed by absent teammates, team capped at 240 expected minutes, back-to-back and blowout trims) × matchup multipliers (Vegas implied total for pts/ast; pace and opponent rebounding / ball security / turnover forcing for the rest). De-vigged props are blended in per stat; Databallr WOWY per-36 shifts added when supplied. A ridge (α=200) on within-slate z-scores, trained on the structural model's own out-of-sample residuals, nudges the result (capped at ±4). Players with no box-score history fall back to FanDuel FPPG. |
| Validation | `projections.walk_forward_backtest` | Expanding window by date; every input (team factors, rates, ridge) comes from earlier games only. Absent rotation players stand in for the injury report. Also fits the projection-spread curve `sd = a + b·mu`. |
| Simulation | `simulator.py` | Gaussian copula with Gamma marginals; same-game pace/shootout, PG↔big assist synergy, high-usage cannibalisation, starter/backup swap and close-game bring-back correlations, optionally blended with empirical correlations. |
| Optimizer | `optimizer.py` | HiGHS MILP (`scipy.optimize.milp`): 2 PG / 2 SG / 2 SF / 2 PF / 1 C, $60,000 cap, ≤4 per team, ≥2 games. Candidates are solved per simulated draw with an ownership-leverage term, scored against a simulated field, and selected greedily for portfolio EV under hard `max_exposure` and `min_unique_players` limits. |
| Ownership | `data_sources.estimate_slate_ownership` | External CSV where names match; otherwise a value/projection softmax heuristic scaled to 900% total. The heuristic is not fit to real contest ownership. |

## Walk-forward results on real data

`results/walk_forward_cv_2026-11-21.json`: 2026-27 regular season through Nov 21, 25 evaluation dates, 3,958 player-games
(every player who played), team factors re-derived from prior games each date.

| Model | MAE | RMSE | Bias | Spearman (per slate) | Top-10% capture |
|---|---|---|---|---|---|
| Season FanDuel average | 7.61 | 9.83 | −0.49 | 0.725 | 0.484 |
| Structural | **7.52** | **9.61** | +0.77 | 0.743 | 0.465 |
| Structural + ridge | 7.52 | 9.61 | +0.77 | **0.745** | 0.457 |

- The structural model beats the season average on error and on overall ranking, but captures the actual top 10% of scorers
  slightly less often. The ridge adds essentially nothing yet (three weeks of data).
- Two settings were chosen on this same backtest, so these numbers are mildly optimistic:
  `shrink_k_minutes=40` (240 under-projected 45+ FPPG players by ~6 points) and `redistribute_share=0.6` (0.85 over-projected
  teammates of absent players).
- The backtest knows who actually played (absent rotation players are treated as ruled out), which is roughly what a final
  injury report gives you before lock — not more.

## Known limits

- **Top-1% / top-0.1% rates and GPP EV are relative scores.** The field is 35 optimizer-built "chalk" lineups, not a real
  contest; use them to rank lineups, not as win probabilities. Calibrate against real contest results before trusting levels.
- **Ownership** is a heuristic unless you pass `--ownership-csv`.
- **Untested against live services from this sandbox** (hosts not reachable here): Databallr live scrape, official injury-PDF
  download, The Odds API. Their parsers are unit-tested on fixtures; the Databallr scrape is best-effort and the CSV export is
  the reliable path.
- ESPN sometimes posts no line for a game; those teams fall back to pace/defence multipliers.

## Outputs

- `fanduel_upload.csv` — FanDuel Classic upload (`PG,PG,SG,SG,SF,SF,PF,PF,C`, cells `id:name`).
- `portfolio_summary.csv` — per-lineup salary, mean, p90, p99, top-1% / top-0.1% rates, GPP EV, ownership, main game stack.
- `exposure_report.csv` — player exposure vs projected ownership.
- `slate_projections.csv` — player projections, minutes, components, matchup features, ridge adjustment, ownership.
- `walk_forward_cv.json` — out-of-sample metrics for the logs supplied.

## Changes to the uploaded files

- `optimizer.py`: the selection loop ran `min(n_lineups, len(candidates))` times, so a small candidate pool silently returned
  too few lineups; and its fallback relaxed `max_exposure` / `min_unique_players`. It now keeps both limits hard and solves
  extra candidates (excluding capped players, with an escalating diversity penalty) when the pool runs dry.
- `run_pipeline.py`: ridge is fit on out-of-sample structural residuals from the walk-forward (instead of residuals from a
  plain season average, which would double-count matchup effects the projection already prices); logs are cut at the slate
  date; ESPN-derived team factors and ESPN lines are used when Databallr/Odds inputs are absent; the demo is labelled as
  synthetic (its team ratings were described as a real Databallr snapshot, which I could not verify); `build_parser()` split out.
- `config.py`, `normalize.py`, `data_sources.py`, `projections.py` and `tests/test_components.py` were written here to match the
  interfaces the uploaded files and `tests/test_pipeline.py` expect.
