"""Date windows for harvesting 'perovskite'-titled papers at <=200 results/call."""
import calendar, json

def month_windows(y, halves=False):
    out = []
    for m in range(1, 13):
        last = calendar.monthrange(y, m)[1]
        if halves:
            out += [(f"{y}-{m:02d}-01", f"{y}-{m:02d}-15"), (f"{y}-{m:02d}-16", f"{y}-{m:02d}-{last}")]
        else:
            out.append((f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last}"))
    return out

W = [("1900-01-01", "1994-12-31"), ("1995-01-01", "1997-12-31"), ("1998-01-01", "2000-12-31"),
     ("2001-01-01", "2002-12-31"), ("2003-01-01", "2004-12-31")]
W += [(f"{y}-01-01", f"{y}-12-31") for y in range(2005, 2013)]
for y in (2013, 2014):
    W += [(f"{y}-01-01", f"{y}-03-31"), (f"{y}-04-01", f"{y}-06-30"), (f"{y}-07-01", f"{y}-09-30"), (f"{y}-10-01", f"{y}-12-31")]
W += [("2015-01-01", "2015-02-28"), ("2015-03-01", "2015-04-30"), ("2015-05-01", "2015-06-30"),
      ("2015-07-01", "2015-08-31"), ("2015-09-01", "2015-10-31"), ("2015-11-01", "2015-12-31")]
for y in range(2016, 2020):
    W += month_windows(y)
for y in range(2020, 2027):
    W += month_windows(y, halves=True)
W = [w for w in W if w[0] < "2026-10-01"]
W.append(("2026-10-01", "2026-10-31"))   # last partial month: half-month window too small to spill to file
json.dump(W, open("data/valency/harvest_windows.json", "w"))
print(len(W))
