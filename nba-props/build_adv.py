# Player-game table with the raw counting stats needed for advanced metrics, plus team/opponent totals
import json, sys, pandas as pd
D=sys.argv[1]; ev=json.load(open(f"{D}/events.json")); rows=[]
def mk(s):
    m,a=s.split("-"); return int(m),int(a)
for eid,meta in ev.items():
    if meta["status"]!="STATUS_FINAL": continue
    s=json.load(open(f"{D}/sum_{eid}.json")); comp=s["header"]["competitions"][0]
    teams={c["homeAway"]:c["team"]["abbreviation"] for c in comp["competitors"]}
    pc=(s.get("pickcenter") or [{}])[0]
    for tb in s["boxscore"].get("players",[]):
        team=tb["team"]["abbreviation"]; st=tb["statistics"][0]; k=st["keys"]
        for ath in st["athletes"]:
            if ath.get("didNotPlay") or not ath.get("stats"): continue
            v=dict(zip(k,ath["stats"]))
            try: mins=float(v["minutes"])
            except: continue
            if mins<=0: continue
            fgm,fga=mk(v["fieldGoalsMade-fieldGoalsAttempted"]); tpm,tpa=mk(v["threePointFieldGoalsMade-threePointFieldGoalsAttempted"]); ftm,fta=mk(v["freeThrowsMade-freeThrowsAttempted"])
            rows.append(dict(eid=eid,date=meta["date"],pid=ath["athlete"]["id"],name=ath["athlete"]["displayName"],
              pos=ath["athlete"].get("position",{}).get("abbreviation"),team=team,
              opp=teams["away"] if team==teams["home"] else teams["home"],home=int(team==teams["home"]),
              starter=int(bool(ath.get("starter"))),min=mins,pts=int(v["points"]),reb=int(v["rebounds"]),
              oreb=int(v["offensiveRebounds"]),dreb=int(v["defensiveRebounds"]),ast=int(v["assists"]),tov=int(v["turnovers"]),
              fgm=fgm,fga=fga,tpm=tpm,tpa=tpa,ftm=ftm,fta=fta,total=pc.get("overUnder"),spread=pc.get("spread")))
pg=pd.DataFrame(rows)
tm=pg.groupby(["eid","team"])[["min","pts","reb","oreb","dreb","ast","tov","fgm","fga","tpm","fta"]].sum().add_prefix("t_").reset_index()
op=tm.rename(columns={c:c.replace("t_","o_") for c in tm if c.startswith("t_")}).rename(columns={"team":"opp"})
pg=pg.merge(tm,on=["eid","team"]).merge(op,on=["eid","opp"])
pg["t_poss"]=pg.t_fga+0.44*pg.t_fta-pg.t_oreb+pg.t_tov; pg["o_poss"]=pg.o_fga+0.44*pg.o_fta-pg.o_oreb+pg.o_tov
pg.to_csv(f"{D}/player_games_adv.csv",index=False); print(pg.shape, pg.pos.value_counts().to_dict())
