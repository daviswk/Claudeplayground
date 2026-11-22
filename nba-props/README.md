# NBA player props — 2026-11-13 (POR @ SAC, OKC @ LAC)

Method adapted from repos under the GitHub `nba-prediction` topic:
- **KengoA/fantasy-basketball** — per-player rolling/weighted-average + volatility features, LightGBM regressor per stat.
- **kyleskom/NBA-Machine-Learning-Sports-Betting** — Vegas total/spread as game context; EV formula from `src/Utils/Expected_Value.py`.
- **NBA-Betting/NBA_AI** — same idea of ESPN as the injury/odds source.

Data: ESPN public API (scoreboard + game summaries), 2026-27 regular season through 2026-11-13.
stats.nba.com is unreachable from the cloud sandbox, so the repos' NBA-API scrapers were not used directly.

Run: `python fetch_espn.py data && python build.py data && python model.py data && python edges.py data`
(prop lines are hard-coded in `edges.py` from public previews; update them from your book.)

Caveat: on a 7-day holdout the model roughly ties a season-average baseline (see `results/model_eval.json`).

## v2 — advanced-stats model with lineups (`build_adv.py`, `model_adv.py`, `tonight_adv.py`)
Projection = projected minutes × season-long per-play rates (USG%, points per play / TS%, REB%, AST%, 3PA rate, 3P%)
× game pace × opponent factors × Vegas total, plus redistribution of absent players' usage and rebounds.
- Rates are season-long (no recency weighting), shrunk toward position averages with a 240-minute prior.
- Minutes come from the player's average in tonight's role (starter or bench), with a blowout trim for starters when the spread is above 4.
- Lineups: SAC confirmed (NBC Sports); POR = last game's starters; OKC/LAC projected. Players absent from the team's last 4 games are treated as out.
- Backtest (last 7 days, actual starters): roughly ties season averages (see `results/adv_eval.json`).

## FanDuel GPP builder
`../nba-fanduel-gpp/` turns these ESPN logs into FanDuel NBA Classic tournament lineups (projections, correlated simulation,
MILP portfolio). `fetch_espn.py` now takes an end date: `python fetch_espn.py OUT_DIR [YYYY-MM-DD]`.
