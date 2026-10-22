"""Attach literature synthesis routes to top-scoring unlabeled candidates.

For each candidate composition: abstracts that mention it in a synthesis
context (direct evidence); otherwise abstracts for its nearest
literature-synthesized neighbours in element-fraction space (analogy).
Routes are tagged by keyword; the cited papers are listed so the full recipe
can be pulled (e.g. via Scite full text) for the ones worth trying.

usage: python3 src/recipes.py "<score column in results/valency_oob_scores.csv>" [top_k]
"""
import gzip
import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from composition import parse_name  # noqa: E402
from lit_layers import frac_vector  # noqa: E402

ROUTES = {
    "hot-injection": r"hot[- ]injection",
    "LARP / reprecipitation": r"reprecipitation|\bLARP\b",
    "antisolvent / spin-coating": r"anti-?solvent|spin[- ]coat",
    "solvothermal / hydrothermal": r"solvothermal|hydrothermal",
    "solid-state / mechanochemical": r"solid[- ]state (reaction|synthesis|route)|ball[- ]mill|mechanochemi|grinding",
    "vapor deposition": r"vapou?r[- ](deposition|transport)|thermal(ly)? evaporat|\bCVD\b|\bPLD\b|pulsed laser|co-?evaporat|sputter",
    "melt / Bridgman growth": r"bridgman|melt[- ]grow|from the melt|zone[- ]melting",
    "solution crystal growth": r"inverse temperature crystalli[sz]ation|slow evaporation|slow cooling|cooling crystalli[sz]ation|single crystals? (were |was )?grown",
    "sulfurization / selenization": r"sulfuri[sz]|selenri?[sz]|selenization|\bCS2\b|H2S",
    "high pressure": r"high[- ]pressure synthesis|high pressure",
    "precipitation / room temperature": r"room[- ]temperature (synthesis|precipitation)|co-?precipitation|precipitat",
}
ROUTE_RE = {k: re.compile(v, re.I) for k, v in ROUTES.items()}


def load_corpus():
    with gzip.open("data/valency/corpus.jsonl.gz", "rt") as fh:
        return {p["id"]: p for p in map(json.loads, fh)}


def routes_of(text):
    return [k for k, r in ROUTE_RE.items() if r.search(text or "")]


def main(score_col, top_k=25):
    S = pd.read_csv("results/valency_oob_scores.csv")
    pos_keys = set(S.comp_key[S.y == 1])
    U = S[(S.y == 0) & ~S.comp_key.isin(pos_keys)]
    best = U.sort_values(score_col, ascending=False).drop_duplicates("comp_key").head(top_k)

    M = pd.read_csv("data/valency/mentions.csv")
    M = M[M.synth_context]
    corpus = load_corpus()
    lit = M.groupby("comp_key").agg(formula=("formula_raw", "first"), papers=("paper_id", lambda s: list(dict.fromkeys(s))))
    lit = lit[[_ok(f) for f in lit.formula]]
    species = sorted({s for f in list(lit.formula) + list(best.Name) for s in parse_name(f)})
    R = np.vstack([frac_vector(f, species) for f in lit.formula])
    R /= np.linalg.norm(R, axis=1, keepdims=True)

    rows, md = [], ["# Candidate synthesis routes from the Valency corpus\n",
                    f"Model: `{score_col}`; top {top_k} unlabeled compositions with no synthesized polymorph in final_data.csv.\n"]
    for _, c in best.iterrows():
        if c.comp_key in lit.index:
            evidence, srcs = "direct", [(lit.loc[c.comp_key, "formula"], 1.0, lit.loc[c.comp_key, "papers"])]
        else:
            v = frac_vector(c.Name, species)
            sim = R @ (v / np.linalg.norm(v))
            idx, seen = [], []
            for i in np.argsort(sim)[::-1]:   # nearest neighbours with distinct element sets
                sp = set(parse_name(lit.formula.iloc[i]))
                if sp not in seen:
                    idx.append(i)
                    seen.append(sp)
                if len(idx) == 3:
                    break
            evidence = "analogy"
            srcs = [(lit.formula.iloc[i], float(sim[i]), lit.papers.iloc[i]) for i in idx]
        cnt, refs = Counter(), []
        for formula, sim, pids in srcs:
            for pid in pids[:40]:
                p = corpus.get(pid, {})
                r = routes_of(f"{p.get('title', '')} {p.get('abstract', '')}")
                cnt.update(r)
                if r and len(refs) < 4:
                    refs.append(f"[{formula}] {p.get('title', '')[:110]} ({str(p.get('date', ''))[:4]}) {p.get('url', '')}")
        rows.append({"Name": c.Name, "structure": c["Crystal Structure"], "source": c.source,
                     "score": round(float(c[score_col]), 3), "evidence": evidence,
                     "neighbours": "; ".join(f"{f} ({s:.2f})" for f, s, _ in srcs),
                     "routes": "; ".join(f"{k} x{n}" for k, n in cnt.most_common(4)),
                     "references": " | ".join(refs)})
        md.append(f"## {c.Name} ({c['Crystal Structure']}), score {c[score_col]:.3f}, {evidence}\n")
        md.append(f"- neighbours: {rows[-1]['neighbours']}\n- routes: {rows[-1]['routes'] or 'none tagged'}\n")
        md += [f"  - {r}\n" for r in refs]
    pd.DataFrame(rows).to_csv("results/candidate_routes.csv", index=False)
    open("results/candidate_routes.md", "w").write("\n".join(md))
    print(pd.DataFrame(rows)[["Name", "structure", "score", "evidence", "routes"]].to_string())


def _ok(f):
    try:
        parse_name(f)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25)
