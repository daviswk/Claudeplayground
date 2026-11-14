import sys, json, pandas as pd, numpy as np
from scipy.stats import norm
D=sys.argv[1]; P=pd.read_csv(f"{D}/tonight_proj.csv",dtype={"pid":str}); ev=json.load(open(f"{D}/model_eval.json"))
pg=pd.read_csv(f"{D}/player_games.csv",dtype={"pid":str}); pg["pra"]=pg.pts+pg.reb+pg.ast; pg["pa"]=pg.pts+pg.ast
P["proj_pa"]=P.proj_pts+P.proj_ast; ev["pa"]=ev["pts"]
L=[("Deni Avdija","pts",23.5),("Deni Avdija","reb",6.5),("Deni Avdija","ast",6.5),("Deni Avdija","pa",30.5),
   ("Damian Lillard","pts",21.5),("Damian Lillard","reb",3.5),("Damian Lillard","ast",4.5),
   ("Donovan Clingan","pts",10.5),("Donovan Clingan","reb",11.5),("Donovan Clingan","ast",2.5),
   ("Jrue Holiday","pts",11.5),("Jrue Holiday","reb",3.5),("Jrue Holiday","ast",3.5),
   ("Darius Acuff Jr.","pts",24.5),("Darius Acuff Jr.","reb",3.5),("Darius Acuff Jr.","ast",6.5),
   ("Zach LaVine","pts",18.5),("Zach LaVine","reb",2.5),("Zach LaVine","ast",2.5),
   ("Domantas Sabonis","pts",12.5),("Domantas Sabonis","reb",11.5),("Domantas Sabonis","ast",5.5),
   ("Keegan Murray","pra",16.5),("Shai Gilgeous-Alexander","pts",31.5)]
out=[]
for nm,s,line in L:
    r=P[P.name==nm].iloc[0]; proj=r[f"proj_{s}"]; sd=ev[s]["resid_sd_per_sqrt"]*np.sqrt(max(proj,1))
    po=1-norm.cdf(line,proj,sd); h=pg[pg.name==nm][s]
    side="OVER" if po>0.5 else "UNDER"; p=max(po,1-po)
    hit=(h>line).mean() if side=="OVER" else (h<line).mean()
    avg=h.mean(); agree=(avg>line)==(side=="OVER")
    out.append(dict(player=nm,team=r.team,stat=s.upper(),line=line,proj=round(proj,1),season_avg=round(avg,1),l5=round(h.tail(5).mean(),1),
        side=side,model_prob=round(p,3),hit_rate=f"{int(round(hit*len(h)))}/{len(h)}",hit=hit,agree=agree,
        ev_110=round(p*90.91-(1-p)*100,1)))
E=pd.DataFrame(out); E["score"]=E.model_prob*0.6+E.hit*0.4
E=E.sort_values("score",ascending=False); E.to_csv(f"{D}/edges.csv",index=False)
pd.set_option("display.width",250); print(E.drop(columns=["hit"]).to_string(index=False))
