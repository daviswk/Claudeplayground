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
