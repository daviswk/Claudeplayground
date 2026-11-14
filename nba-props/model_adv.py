"""Rate-based ("advanced stats") projection: minutes x shrunk per-play rates x pace x opponent, plus usage redistribution."""
import sys, json, pandas as pd, numpy as np
D=sys.argv[1]; pg=pd.read_csv(f"{D}/player_games_adv.csv",dtype={"pid":str,"eid":str,"date":str})
pg["dt"]=pd.to_datetime(pg.date); pg["plays"]=pg.fga+0.44*pg.fta+pg.tov; pg["t_plays"]=pg.t_fga+0.44*pg.t_fta+pg.t_tov
pg["pos"]=pg.pos.fillna("F")
K_MIN=240  # minutes of league-average prior blended into each player's rates

def rates(h):
    """Season-long advanced rates per player from history h (no recency weighting)."""
    tg=h.drop_duplicates(["eid","team"])
    team=tg.groupby("team").agg(t_min=("t_min","sum"),t_plays=("t_plays","sum"),t_poss=("t_poss","sum"),o_poss=("o_poss","sum"),
        t_fgm=("t_fgm","sum"),t_pts=("t_pts","sum"),o_pts=("o_pts","sum"),reb_env=("t_reb","sum"),o_reb=("o_reb","sum"),g=("eid","count"))
    team["pace"]=48*((team.t_poss+team.o_poss)/2)/(team.t_min/5)
    team["plays48"]=48*team.t_plays/(team.t_min/5); team["fgm48"]=48*team.t_fgm/(team.t_min/5)
    team["drtg"]=100*team.o_pts/team.o_poss; team["ortg"]=100*team.t_pts/team.t_poss
    team["reb48_allowed_env"]=48*(team.reb_env+team.o_reb)/(team.t_min/5)
    # opponent-allowed environment (as the opponent)
    opp=tg.groupby("opp").agg(o_plays=("t_plays","sum"),pts=("t_pts","sum"),min=("t_min","sum"),ast=("t_ast","sum"),fgm=("t_fgm","sum"),
        reb=("t_reb","sum"),oreb=("o_reb","sum"),tpm=("t_tpm","sum"))
    lg=dict(ppp=opp.pts.sum()/opp.o_plays.sum(), ast_fgm=opp.ast.sum()/opp.fgm.sum(), tpm_play=opp.tpm.sum()/opp.o_plays.sum(),
            reb48=48*(opp.reb.sum()+opp.oreb.sum())/(opp["min"].sum()/5), pace=team.pace.mean())
    shrink=lambda x,lgv,w: lgv+(x-lgv)*w
    wopp=(opp.o_plays/(opp.o_plays+1500)).clip(0,1)   # ~15 games of plays before trusting fully
    opp["f_ppp"]=shrink(opp.pts/opp.o_plays/lg["ppp"],1,wopp); opp["f_ast"]=shrink(opp.ast/opp.fgm/lg["ast_fgm"],1,wopp)
    opp["f_3"]=shrink(opp.tpm/opp.o_plays/lg["tpm_play"],1,wopp)
    opp["f_reb"]=shrink(48*(opp.reb+opp.oreb)/(opp["min"]/5)/lg["reb48"],1,wopp)
    p=h.groupby("pid").agg(name=("name","last"),team=("team","last"),pos=("pos","last"),min=("min","sum"),g=("eid","count"),
        pts=("pts","sum"),reb=("reb","sum"),ast=("ast","sum"),tpm=("tpm","sum"),tpa=("tpa","sum"),fga=("fga","sum"),fta=("fta","sum"),
        plays=("plays","sum"),fgm=("fgm","sum"))
    # on-court team context, minute-weighted
    h=h.assign(w_tplays=h.t_plays*h["min"]/(h.t_min/5), w_reb=(h.t_reb+h.o_reb)*h["min"]/(h.t_min/5), w_fgm=(h.t_fgm)*h["min"]/(h.t_min/5))
    oc=h.groupby("pid")[["w_tplays","w_reb","w_fgm"]].sum(); p=p.join(oc)
    p["usg"]=p.plays/p.w_tplays; p["ts"]=p.pts/(2*(p.fga+0.44*p.fta)); p["ppp"]=p.pts/p.plays
    p["reb_pct"]=p.reb/p.w_reb; p["ast_pct"]=p.ast/(p.w_fgm-p.fgm); p["tpa_play"]=p.tpa/p.plays
    p["tp_pct"]=(p.tpm+0.36*60)/(p.tpa+60)           # 3P% shrunk with 60 attempts at 36%
    for c in ["usg","ppp","reb_pct","ast_pct","tpa_play"]:
        pri=p.groupby("pos").apply(lambda d:(d[c]*d["min"]).sum()/d["min"].sum())
        p[c]=(p[c]*p["min"]+p.pos.map(pri)*K_MIN)/(p["min"]+K_MIN)
    # minutes by role
    rm=h.groupby(["pid","starter"])["min"].agg(["mean","count"]).unstack()
    p["min_start"]=rm.get(("mean",1)); p["n_start"]=rm.get(("count",1)); p["min_bench"]=rm.get(("mean",0)); p["n_bench"]=rm.get(("count",0))
    p["min_avg"]=p["min"]/p.g
    p["gp_frac"]=p.g/p.team.map(team.g)
    return p, team, opp, lg

def project(p, team, opp, lg, tm, op, spread, total, starters, out_names, k_usage=0.7):
    """Project one team tm vs op. spread = tm's spread (negative = favourite)."""
    act=p[(p.team==tm)&(~p.name.isin(out_names))].copy()
    act=act[act.g>=2]
    # minutes by tonight's role, shrunk toward overall average
    def mins(r):
        st=r["name"] in starters
        m,n=(r.min_start,r.n_start) if st else (r.min_bench,r.n_bench)
        base=r.min_avg if pd.isna(m) else (m*n+r.min_avg*2)/(n+2)
        if st and pd.isna(r.min_start): base=max(base,26)   # new starter with no starts: floor
        return base
    act["pmin"]=act.apply(mins,axis=1)
    # blowout: starters lose ~0.12 min per point of spread beyond 4
    blow=max(abs(spread)-4,0)*0.12
    act.loc[act.name.isin(starters),"pmin"]-=blow
    bench=~act.name.isin(starters)
    act.loc[bench,"pmin"]*=act.loc[bench,"gp_frac"].clip(upper=1)        # bench: weight by how often they play
    over=act.pmin.sum()-240
    if over>0:                                                             # trim bench first, then everyone
        bs=act.loc[bench,"pmin"].sum(); cut=min(over,bs*0.35)
        act.loc[bench,"pmin"]*=1-cut/bs; over-=cut
        if over>0: act["pmin"]*=240/act.pmin.sum()
    # usage redistribution from tonight's absences (only the portion they were present for in the rate sample)
    outp=p[(p.team==tm)&(p.name.isin(out_names))]
    miss=(outp.usg*outp.min_avg/48*outp.gp_frac).sum()   # share of team plays missing
    act["usg_adj"]=act.usg*(1+ k_usage*miss*(act.usg*act.pmin).sum()/(act.usg*act.pmin).sum()/ (act.usg*act.pmin/48).sum())
    miss_reb=(outp.reb_pct*outp.min_avg/48*outp.gp_frac).sum()
    act["reb_adj"]=act.reb_pct*(1+k_usage*miss_reb/((act.reb_pct*act.pmin/48).sum()))
    # game pace and environment (Vegas total folds in pace + efficiency, kyleskom-style)
    gpace=team.loc[tm,"pace"]*team.loc[op,"pace"]/lg["pace"]
    tot_f=total/ (team.loc[tm,"ortg"]*gpace/100 + team.loc[op,"ortg"]*gpace/100)
    plays48=team.loc[tm,"plays48"]*gpace/team.loc[tm,"pace"]
    of=opp.loc[op]
    act["plays"]=act.pmin/48*plays48*act.usg_adj
    act["p_pts"]=act.plays*act.ppp*of.f_ppp*(0.5+0.5*tot_f)*(1-0.15*(act.usg_adj/act.usg-1))  # small efficiency tax on extra usage
    act["p_tpm"]=act.plays*act.tpa_play*act.tp_pct*of.f_3
    act["p_reb"]=act.pmin/48*lg["reb48"]*gpace/lg["pace"]*of.f_reb*act.reb_adj
    fgm48=team.loc[tm,"fgm48"]*gpace/team.loc[tm,"pace"]*(0.5+0.5*tot_f)
    act["p_ast"]=act.pmin/48*fgm48*(1-act.usg_adj)*act.ast_pct*of.f_ast*(act.usg_adj/act.usg)**0.5
    act["p_pra"]=act.p_pts+act.p_reb+act.p_ast
    return act

if __name__=="__main__":
    mode=sys.argv[2] if len(sys.argv)>2 else "tonight"
    if mode=="backtest":
        cut=pg.dt.max()-pd.Timedelta(days=7); hist=pg[pg.dt<=cut]; test=pg[pg.dt>cut]
        p,team,opp,lg=rates(hist); errs=[]
        for (eid,tmn),gdf in test.groupby(["eid","team"]):
            r=gdf.iloc[0]; sp=r.spread if r.home else -r.spread
            if pd.isna(sp) or tmn not in team.index or r.opp not in team.index: continue
            played=set(gdf.name); outn=set(p[(p.team==tmn)&(~p.name.isin(played))].name)
            a=project(p,team,opp,lg,tmn,r.opp,sp,r.total,set(gdf[gdf.starter==1].name),outn)
            m=a.merge(gdf[["pid","pts","reb","ast","tpm","min"]].rename(columns={"pts":"a_pts","reb":"a_reb","ast":"a_ast","tpm":"a_tpm","min":"a_min"}),left_index=True,right_on="pid")
            m=m.merge(hist.groupby("pid")[["pts","reb","ast","tpm"]].mean().add_suffix("_savg"),left_on="pid",right_index=True)
            errs.append(m)
        E=pd.concat(errs); E=E[E.a_min>=10]
        out={}
        for s in ["pts","reb","ast","tpm"]:
            out[s]=dict(bias=round((E[f"p_{s}"]-E[f"a_{s}"]).mean(),2), adv_mae=round((E[f"p_{s}"]-E[f"a_{s}"]).abs().mean(),2), season_avg_mae=round((E[f"{s}_savg"]-E[f"a_{s}"]).abs().mean(),2),
                        resid_sd_per_sqrt=float(((E[f"a_{s}"]-E[f"p_{s}"])/np.sqrt(E[f"p_{s}"].clip(lower=1))).std()))
        E["pra"]=E.a_pts+E.a_reb+E.a_ast; out["pra"]=dict(adv_mae=round((E.p_pra-E.pra).abs().mean(),2),
            season_avg_mae=round((E.pts_savg+E.reb_savg+E.ast_savg-E.pra).abs().mean(),2),resid_sd_per_sqrt=float(((E.pra-E.p_pra)/np.sqrt(E.p_pra.clip(lower=1))).std()))
        print(json.dumps(out,indent=1), len(E)); json.dump(out,open(f"{D}/adv_eval.json","w"))
