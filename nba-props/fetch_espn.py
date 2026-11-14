import json, sys, os, datetime as dt, concurrent.futures as cf, urllib.request
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
B = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
def get(u):
    with urllib.request.urlopen(u, timeout=30) as r: return json.load(r)
start, end = dt.date(2026,10,15), dt.date(2026,11,13)
days = [(start+dt.timedelta(d)).strftime("%Y%m%d") for d in range((end-start).days+1)]
events = {}
with cf.ThreadPoolExecutor(8) as ex:
    for d, sb in zip(days, ex.map(lambda d: get(f"{B}/scoreboard?dates={d}"), days)):
        for e in sb.get("events", []):
            if e["season"]["type"] in (2,):  # regular season
                events[e["id"]] = dict(date=d, name=e["shortName"], status=e["status"]["type"]["name"], tip=e["date"])
json.dump(events, open(f"{OUT}/events.json","w"), indent=1)
print(len(events), "events")
def summ(eid):
    p = f"{OUT}/sum_{eid}.json"
    if os.path.exists(p) and events[eid]["status"]=="STATUS_FINAL": return
    json.dump(get(f"{B}/summary?event={eid}"), open(p,"w"))
with cf.ThreadPoolExecutor(10) as ex: list(ex.map(summ, events))
print("done")
