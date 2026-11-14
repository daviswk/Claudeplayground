# KengoA/fantasy-basketball-style rolling features + LightGBM, with kyleskom-style Vegas context
import sys, json, pandas as pd, numpy as np, lightgbm as lgb
D = sys.argv[1]
pg = pd.read_csv(f"{D}/player_games.csv", dtype={"pid":str,"eid":str,"date":str})
g = pd.read_csv(f"{D}/games.csv", dtype={"eid":str,"date":str})
pg["pra"] = pg.pts+pg.reb+pg.ast
STATS = ["pts","reb","ast","tpm","pra"]
pg["dt"] = pd.to_datetime(pg.date); pg = pg.sort_values(["pid","dt"])
def team_ctx(df):
    tsp = np.where(df.home==1, df.spread, -df.spread)
    df["team_spread"] = tsp; df["implied"] = df.total/2 - tsp/2
    return df
pg = team_ctx(pg)
# opponent allowed per stat to each position-agnostic player-minute (expanding, prior only)
og = pg.groupby(["opp","eid","dt"])[["pts","reb","ast","tpm","min"]].sum().reset_index().sort_values("dt")
for s in ["pts","reb","ast","tpm"]:
    og[f"opp_{s}_pm"] = og.groupby("opp")[s].transform(lambda x: x.shift().expanding().sum()) / og.groupby("opp")["min"].transform(lambda x: x.shift().expanding().sum())
def feats(df):
    gp = df.groupby("pid")
    df["gp_prior"] = gp.cumcount()
    df["rest"] = gp.dt.diff().dt.days.clip(upper=5).fillna(3)
    for c in STATS+["min","fga","starter"]:
        sh = gp[c].shift()
        df[f"{c}_avg"] = sh.groupby(df.pid).transform(lambda x: x.expanding().mean())
        df[f"{c}_l5"] = sh.groupby(df.pid).transform(lambda x: x.ewm(span=5).mean())
        df[f"{c}_sd"] = sh.groupby(df.pid).transform(lambda x: x.rolling(10,min_periods=3).std())
    for s in ["pts","reb","ast","tpm"]:
        df[f"{s}_pm"] = df[f"{s}_l5"]/df.min_l5.clip(lower=1)
    return df
pg = feats(pg)
pg = pg.merge(og[["opp","eid"]+[c for c in og if c.startswith("opp_")]], on=["opp","eid"], how="left")
FEAT = [c for c in pg if c.endswith(("_avg","_l5","_sd","_pm"))] + ["gp_prior","rest","home","total","team_spread","implied"]
train = pg[pg.gp_prior>=2].copy()
# time-based validation: last 7 days held out, to calibrate residual spread
cut = train.dt.max() - pd.Timedelta(days=7)
res = {}
models = {}
for s in STATS:
    p = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    tr, va = train[train.dt<=cut], train[train.dt>cut]
    m = lgb.train(p, lgb.Dataset(tr[FEAT], tr[s]), 400)
    pv = m.predict(va[FEAT]); base = va[f"{s}_avg"]
    mae, bmae = np.mean(np.abs(pv-va[s])), np.mean(np.abs(base-va[s]))
    resid = (va[s]-pv)
    res[s] = dict(mae=round(mae,2), season_avg_mae=round(bmae,2), resid_sd_per_sqrt=float(np.std(resid/np.sqrt(np.clip(pv,1,None)))))
    models[s] = lgb.train(p, lgb.Dataset(train[FEAT], train[s]), 400)
print(json.dumps(res, indent=1))
# ---- tonight ----
from scipy.stats import norm
inj = {}
for eid in ["401909303","401909304"]:
    sm = json.load(open(f"{D}/sum_{eid}.json"))
    for t in sm.get("injuries",[]):
        for i in t.get("injuries",[]): inj[i["athlete"]["displayName"]] = i["status"]
out = []
last = pg.sort_values("dt").groupby("pid").tail(1)
for _, gm in g[g.eid.isin(["401909303","401909304"])].iterrows():
    for team, opp, home in [(gm.home, gm.away, 1), (gm.away, gm.home, 0)]:
        roster = last[(last.team==team) & (last.dt >= pg.dt.max()-pd.Timedelta(days=10))]
        allp = pg[pg.pid.isin(roster.pid)]
        for pid, h in allp.groupby("pid"):
            h = h.sort_values("dt"); nm = h.name.iloc[-1]
            if inj.get(nm) in ("Out","Doubtful"): continue
            r = {"pid":pid,"name":nm,"team":team,"opp":opp,"home":home,"gp_prior":len(h),
                 "rest":min((pd.Timestamp("2026-11-13")-h.dt.iloc[-1]).days,5),"total":gm.total,
                 "team_spread": gm.spread if home else -gm.spread}
            r["implied"] = gm.total/2 - r["team_spread"]/2
            for c in STATS+["min","fga","starter"]:
                r[f"{c}_avg"]=h[c].mean(); r[f"{c}_l5"]=h[c].ewm(span=5).mean().iloc[-1]; r[f"{c}_sd"]=h[c].tail(10).std()
            for s in ["pts","reb","ast","tpm"]: r[f"{s}_pm"]=r[f"{s}_l5"]/max(r["min_l5"],1)
            oo = og[og.opp==opp].sort_values("dt")
            for s in ["pts","reb","ast","tpm"]:
                r[f"opp_{s}_pm"] = oo[s].sum()/oo["min"].sum()
            if r["min_l5"] < 20: continue
            X = pd.DataFrame([r])[FEAT]
            for s in STATS: r[f"proj_{s}"] = float(models[s].predict(X)[0])
            r["status"] = inj.get(nm,"Active"); out.append(r)
P = pd.DataFrame(out)
P.to_csv(f"{D}/tonight_proj.csv", index=False)
json.dump(res, open(f"{D}/model_eval.json","w"))
cols = ["name","team","opp","status","min_l5"]+[f"{s}_avg" for s in STATS]+[f"proj_{s}" for s in STATS]
pd.set_option("display.width",250); print(P[cols].round(1).sort_values(["team","proj_pra"],ascending=[True,False]).to_string(index=False))
