"""Copy saved Valency tool outputs into data/valency/raw keyed by query and date window,
then report which planned windows are missing or hit the 200-result cap."""
import glob, json, os, re, sys
T = "/root/.claude/projects/-home-user/3374a572-af78-5a6e-900e-eef094b49e84/tool-results"
OUT = "data/valency/raw"
for f in glob.glob(f"{T}/*.txt"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict) or "papers" not in d:
        continue
    q = d.get("query") or "+".join(d.get("queries", []))
    dr = d.get("date_range") or {}
    key = re.sub(r"[^A-Za-z0-9_.+-]", "-", f"{q}__{dr.get('start') or 'x'}__{dr.get('end') or 'x'}")[:180]
    dst = f"{OUT}/{key}.json"
    if not os.path.exists(dst) or os.path.getsize(dst) < os.path.getsize(f):
        os.system(f"cp '{f}' '{dst}'")
W = json.load(open("data/valency/harvest_windows.json"))
planned = {tuple(w) for w in W}
for f in glob.glob(f"{OUT}/perovskite__*.json"):  # drop windows superseded by finer splits
    dr = json.load(open(f))["date_range"]
    if (dr["start"] or "1900-01-01", dr["end"]) not in planned:
        os.remove(f)
have, capped, n = {}, [], 0
for f in glob.glob(f"{OUT}/perovskite__*.json"):
    d = json.load(open(f)); dr = d["date_range"]
    have[(dr["start"] or "1900-01-01", dr["end"])] = len(d["papers"]); n += len(d["papers"])
missing = [w for w in W if tuple(w) not in have]
capped = [k for k, v in have.items() if v >= 200]
print(f"windows done {len(W) - len(missing)}/{len(W)}  papers(raw)={n}  capped={capped}")
if "-m" in sys.argv:
    print(json.dumps(missing))
