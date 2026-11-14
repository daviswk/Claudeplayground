import sys, json, pandas as pd, numpy as np
from scipy.stats import norm
sys.path.insert(0, sys.argv[2]); from model_adv import pg, rates, project
D=sys.argv[1]; p,team,opp,lg=rates(pg); ev=json.load(open(f"{D}/adv_eval.json"))
last=pg.sort_values("dt").groupby("team").tail(1)
por_last=set(pg[(pg.eid==last[last.team=="POR"].eid.iloc[0])&(pg.team=="POR")&(pg.starter==1)].name)
OUT={"Precious Achiuwa","Maxime Raynaud","Ben Simmons","Chris Youngblood","Vit Krejci","Ja Morant","Shaedon Sharpe",
     "Brandon Ingram","Yanic Konan Niederhauser","Bradley Beal","Jordan Miller","Ajay Mitchell","Thomas Sorber","Jaylin Williams","Cason Wallace","Chet Holmgren"}
recent=pg.sort_values("dt").groupby("team").eid.apply(lambda e: list(dict.fromkeys(e))[-4:])
stale={n for t in ["POR","SAC","OKC","LAC"] for n in set(pg[pg.team==t].name)-set(pg[(pg.team==t)&(pg.eid.isin(recent[t]))].name)}
print("not seen in team's last 4 games -> treated as out:", sorted(stale)); OUT|=stale
ST={"SAC":{"Darius Acuff Jr.","Zach LaVine","De'Andre Hunter","Keegan Murray","Domantas Sabonis"},   # confirmed (NBC Sports)
    "POR":por_last-OUT,                                                                            # last game's starters, minus outs
    "OKC":{"Shai Gilgeous-Alexander","Jared McCain","Alex Caruso","Aday Mara","Isaiah Hartenstein"},  # projected
    "LAC":{"Darius Garland","Kris Dunn","Derrick Jones Jr.","Rui Hachimura","Brook Lopez"}}           # projected
print("POR starters (last game, minus outs):", ST["POR"])
games=[("SAC","POR",10.5,229.5),("LAC","OKC",6.5,222.5)]
A=[]
for h,a,sp,tot in games:
    A.append(project(p,team,opp,lg,h,a,sp,tot,ST[h],OUT)); A.append(project(p,team,opp,lg,a,h,-sp,tot,ST[a],OUT))
A=pd.concat(A); A["starter_tonight"]=[n in ST[t] for n,t in zip(A.name,A.team)]
A.to_csv(f"{D}/tonight_adv.csv")
cols=["name","team","starter_tonight","pmin","min_avg","usg","usg_adj","ts","reb_pct","ast_pct","p_pts","p_reb","p_ast","p_tpm","p_pra"]
pd.set_option("display.width",250)
print(A[A.pmin>=18][cols].sort_values(["team","p_pra"],ascending=[True,False]).round(3).to_string(index=False))
print({t:round(team.loc[t,"pace"],1) for t in ["POR","SAC","OKC","LAC"]}, "lg pace", round(lg["pace"],1))
print({t:{k:round(opp.loc[t,k],3) for k in ["f_ppp","f_reb","f_ast","f_3"]} for t in ["POR","SAC","OKC","LAC"]})
print({t:(round(team.loc[t,"ortg"],1),round(team.loc[t,"drtg"],1)) for t in ["POR","SAC","OKC","LAC"]})
L=[("Deni Avdija","pts",23.5),("Deni Avdija","reb",6.5),("Deni Avdija","ast",6.5),("Damian Lillard","pts",21.5),("Damian Lillard","reb",3.5),("Damian Lillard","ast",4.5),
   ("Donovan Clingan","pts",10.5),("Donovan Clingan","reb",11.5),("Donovan Clingan","ast",2.5),("Jrue Holiday","pts",11.5),("Jrue Holiday","reb",3.5),("Jrue Holiday","ast",3.5),
   ("Darius Acuff Jr.","pts",24.5),("Darius Acuff Jr.","reb",3.5),("Darius Acuff Jr.","ast",6.5),("Zach LaVine","pts",18.5),("Zach LaVine","reb",2.5),("Zach LaVine","ast",2.5),
   ("Domantas Sabonis","pts",12.5),("Domantas Sabonis","reb",11.5),("Domantas Sabonis","ast",5.5),("Keegan Murray","pra",16.5),("Shai Gilgeous-Alexander","pts",31.5)]
rows=[]
for nm,s,line in L:
    r=A[A.name==nm].iloc[0]; pr=r[f"p_{s}"]; sd=ev[s]["resid_sd_per_sqrt"]*np.sqrt(max(pr,1)); po=1-norm.cdf(line,pr,sd)
    side="OVER" if po>.5 else "UNDER"; pp=max(po,1-po)
    rows.append(dict(player=nm,stat=s.upper(),line=line,adv_proj=round(pr,1),pmin=round(r.pmin,1),side=side,prob=round(pp,3),ev_110=round(pp*90.91-(1-pp)*100,1)))
E=pd.DataFrame(rows).sort_values("prob",ascending=False); E.to_csv(f"{D}/edges_adv.csv",index=False); print(E.to_string(index=False))
