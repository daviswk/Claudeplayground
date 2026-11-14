import json, glob, sys, pandas as pd, numpy as np
D = sys.argv[1]
ev = json.load(open(f"{D}/events.json"))
rows, games = [], []
def ctx(s, eid):
    comp = s["header"]["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    pc = (s.get("pickcenter") or [{}])[0]
    total = pc.get("overUnder"); spread = pc.get("spread")  # spread relative to home team
    return teams, total, spread
for eid, meta in ev.items():
    s = json.load(open(f"{D}/sum_{eid}.json"))
    teams, total, spread = ctx(s, eid)
    h, a = teams["home"]["team"]["abbreviation"], teams["away"]["team"]["abbreviation"]
    games.append(dict(eid=eid, date=meta["date"], status=meta["status"], home=h, away=a, total=total, spread=spread))
    if meta["status"] != "STATUS_FINAL": continue
    for tb in s["boxscore"].get("players", []):
        team = tb["team"]["abbreviation"]; st = tb["statistics"][0]; k = st["keys"]
        for ath in st["athletes"]:
            if ath.get("didNotPlay") or not ath.get("stats"): continue
            v = dict(zip(k, ath["stats"]))
            try: mins = float(v["minutes"])
            except: continue
            if mins <= 0: continue
            tpm = int(v["threePointFieldGoalsMade-threePointFieldGoalsAttempted"].split("-")[0])
            fga = int(v["fieldGoalsMade-fieldGoalsAttempted"].split("-")[1])
            rows.append(dict(eid=eid, date=meta["date"], pid=ath["athlete"]["id"], name=ath["athlete"]["displayName"],
                team=team, opp=a if team==h else h, home=int(team==h), starter=int(bool(ath.get("starter"))),
                min=mins, pts=int(v["points"]), reb=int(v["rebounds"]), ast=int(v["assists"]), tpm=tpm, fga=fga,
                total=total, spread=spread))
pg = pd.DataFrame(rows); g = pd.DataFrame(games)
pg.to_csv(f"{D}/player_games.csv", index=False); g.to_csv(f"{D}/games.csv", index=False)
print(pg.shape, g.shape, pg.date.min(), pg.date.max()); print(g[g.date=="20261113"])
