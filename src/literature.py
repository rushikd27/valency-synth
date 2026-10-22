"""Build the literature corpus from harvested Valency results and extract
perovskite formulas mentioned in titles/abstracts, with first-report dates.

Outputs
- data/valency/corpus.jsonl.gz : one record per unique paper (id, date, title, abstract, source)
- data/valency/mentions.csv : (paper_id, date, formula_raw, comp_key, in_title, synth_context)
"""
import glob
import gzip
import json
import re
import sys

import pandas as pd

sys.path.insert(0, "src")
from composition import ANIONS, MOLECULAR, normalize_abx3, parse_name, reduced_key  # noqa: E402

RAW = "data/valency/raw"

SYNTH_WORDS = re.compile(
    r"\b(synthesi[sz]ed|synthesis|prepared|grown|fabricated|crystalli[sz]ed|obtained|hot[- ]injection|"
    r"reprecipitation|solvothermal|hydrothermal|mechanochemical|ball[- ]mill|solid[- ]state reaction|"
    r"single crystals?|nanocrystals?|thin films?|spin[- ]coat|antisolvent|sulfuri[sz]ation|annealed)\b", re.I)
THEORY_WORDS = re.compile(r"first[- ]principles|density functional|\bDFT\b|ab initio|we predict|predicted|"
                          r"high[- ]throughput (screening|calculation)|machine[- ]learning model", re.I)

# formula tokens: element/MA/FA/GA symbols with optional (sub)scripts, >= 2 species, contains an anion
TOKEN = re.compile(r"(?<![A-Za-z])((?:\(?(?:MA|FA|GA|Cs|Rb|K|Na|Li|Ba|Sr|Ca|Pb|Sn|Ge|Bi|Sb|Ag|Cu|Au|In|Tl|Ti|Zr|Hf|"
                   r"Mn|Fe|Co|Ni|Zn|Cd|Hg|Pd|Pt|Y|Sc|La|Ce|Eu|Yb|Al|Ga|As|Mg|Ir|Rh|Mo|W|Ta|Nb|V|Cr|"
                   r"Cl|Br|I|F|S|Se|Te|O|N|H|C|B|P)\)?[\d.xyzδ\-+]*)+)")


def clean_text(s):
    s = s or ""
    s = re.sub(r"<[^>]+>", "", s)                       # stray HTML (<sub>, <i>)
    s = re.sub(r"\$_\{?([\d.]+)\}?\$|_\{?([\d.]+)\}?", lambda m: m.group(1) or m.group(2), s)  # LaTeX subscripts
    s = s.replace("$", "").replace(" ", "")
    s = s.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    s = s.replace("CH3NH3", "MA").replace("(CH3NH3)", "MA").replace("CH(NH2)2", "FA").replace("HC(NH2)2", "FA")
    return s


def candidate_formulas(text):
    out = set()
    for m in TOKEN.finditer(text):
        tok = m.group(1).strip("()-+.,;:")
        if len(tok) < 4 or re.fullmatch(r"[A-Z][a-z]?\d*", tok):
            continue
        try:
            c = parse_name(tok)
        except (ValueError, ZeroDivisionError):
            continue
        species = set(c)
        if len(species) < 2 or not species & set(ANIONS):
            continue
        if "O" in species:          # skip oxides / oxy-anions: out of scope for this dataset
            continue
        if species & {"P", "C", "N", "H", "B"}:   # abbreviations (PSCs, PVSCs) and organics other than MA/FA/GA
            continue
        cations = species - set(ANIONS)
        if len(cations) < 2:        # binaries / fragments: PbI2, PbBr4, Pb-I
            continue
        out.add(tok)
    return out


def load_corpus():
    papers = {}
    for f in glob.glob(f"{RAW}/*.json"):
        d = json.load(open(f))
        for p in d.get("papers", []):
            if p["id"] in papers:
                continue
            fa = (p.get("dates") or {}).get("first_available") or {}
            date = fa.get("value") or fa.get("start") or p.get("first_submitted") or p.get("datestamp")
            papers[p["id"]] = {"id": p["id"], "date": date, "title": p.get("title", ""),
                               "abstract": p.get("abstract", ""), "source": p.get("source"),
                               "doi": p.get("doi", ""), "url": p.get("url", "")}
    return papers


def main():
    papers = load_corpus()
    with gzip.open("data/valency/corpus.jsonl.gz", "wt") as fh:
        for p in papers.values():
            fh.write(json.dumps(p) + "\n")
    rows = []
    for p in papers.values():
        title, abstract = clean_text(p["title"]), clean_text(p["abstract"])
        text = f"{title} {abstract}"
        synth = bool(SYNTH_WORDS.search(text))
        theory = bool(THEORY_WORDS.search(text))
        for tok in candidate_formulas(text):
            try:
                key = str(reduced_key(tok))
            except Exception:
                continue
            rows.append({"paper_id": p["id"], "date": p["date"], "formula_raw": tok, "comp_key": key,
                         "in_title": tok in title, "synth_context": synth and not theory, "theory": theory})
    M = pd.DataFrame(rows)
    M.to_csv("data/valency/mentions.csv", index=False)
    print(f"papers={len(papers)} mentions={len(M)} unique_compositions={M.comp_key.nunique() if len(M) else 0}")


if __name__ == "__main__":
    main()
