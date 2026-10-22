"""mat2vec-style word2vec on the Valency perovskite corpus.

Formulas in abstracts are split into their element symbols and element/anion
names are mapped to symbols, so every species gets one vector learned from the
contexts it appears in. A composition is embedded as the fraction-weighted mean
of its species vectors, so formulas that never appear in the corpus still get a
vector and no composition is looked up by its own formula token.
"""
import gzip
import json
import re
import sys

import numpy as np
from gensim.models import Word2Vec

sys.path.insert(0, "src")
from composition import parse_name  # noqa: E402
from literature import TOKEN, clean_text  # noqa: E402

NAMES = {
    "cesium": "Cs", "caesium": "Cs", "rubidium": "Rb", "potassium": "K", "sodium": "Na", "lithium": "Li",
    "barium": "Ba", "strontium": "Sr", "calcium": "Ca", "magnesium": "Mg", "lead": "Pb", "tin": "Sn",
    "germanium": "Ge", "bismuth": "Bi", "antimony": "Sb", "silver": "Ag", "copper": "Cu", "gold": "Au",
    "indium": "In", "thallium": "Tl", "titanium": "Ti", "zirconium": "Zr", "hafnium": "Hf", "manganese": "Mn",
    "iron": "Fe", "cobalt": "Co", "nickel": "Ni", "zinc": "Zn", "cadmium": "Cd", "mercury": "Hg",
    "palladium": "Pd", "platinum": "Pt", "yttrium": "Y", "scandium": "Sc", "lanthanum": "La", "cerium": "Ce",
    "europium": "Eu", "ytterbium": "Yb", "aluminum": "Al", "aluminium": "Al", "gallium": "Ga", "arsenic": "As",
    "iridium": "Ir", "rhodium": "Rh", "molybdenum": "Mo", "tungsten": "W", "tantalum": "Ta", "niobium": "Nb",
    "vanadium": "V", "chromium": "Cr", "chloride": "Cl", "chlorine": "Cl", "bromide": "Br", "bromine": "Br",
    "iodide": "I", "iodine": "I", "fluoride": "F", "fluorine": "F", "sulfide": "S", "sulphide": "S",
    "sulfur": "S", "selenide": "Se", "selenium": "Se", "telluride": "Te", "tellurium": "Te",
    "methylammonium": "MA", "formamidinium": "FA", "guanidinium": "GA",
}
WORD = re.compile(r"[A-Za-z][A-Za-z\-]*")


def tokenize(text):
    text = clean_text(text)
    out, pos = [], 0
    for m in TOKEN.finditer(text):
        out += _words(text[pos:m.start()])
        tok = m.group(1).strip("()-+.,;:")
        try:
            species = list(parse_name(tok)) if len(tok) >= 3 else []
        except (ValueError, ZeroDivisionError):
            species = []
        out += species if len(species) >= 2 else _words(m.group(0))
        pos = m.end()
    return out + _words(text[pos:])


def _words(s):
    toks = []
    for w in WORD.findall(s):
        lw = w.lower()
        toks.append(NAMES.get(lw, lw))
    return toks


def load_sentences(cutoff=None, path="data/valency/corpus.jsonl.gz"):
    sents = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            p = json.loads(line)
            if cutoff and (not p["date"] or str(p["date"])[:10] >= cutoff):
                continue
            for part in re.split(r"(?<=[.!?])\s+", f"{p['title']}. {p['abstract'] or ''}"):
                t = tokenize(part)
                if len(t) > 3:
                    sents.append(t)
    return sents


def train(cutoff=None, dim=100, seed=0):
    sents = load_sentences(cutoff)
    model = Word2Vec(sents, vector_size=dim, window=8, min_count=3, sg=1, negative=10,
                     epochs=15, workers=4, seed=seed)
    return model, len(sents)


def compose(model, name):
    """Fraction-weighted mean of species vectors (unit-normalized); NaN if a species is unknown."""
    c = parse_name(name)
    tot = sum(c.values())
    v = np.zeros(model.vector_size)
    for s, f in c.items():
        if s not in model.wv:
            return np.full(model.vector_size, np.nan)
        w = model.wv[s]
        v += (f / tot) * w / np.linalg.norm(w)
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    cutoff = sys.argv[1] if len(sys.argv) > 1 else None
    model, n = train(cutoff)
    tag = cutoff or "all"
    model.save(f"data/valency/w2v_{tag}.model")
    print(f"cutoff={tag} sentences={n} vocab={len(model.wv)}")
    for w in ["Cs", "Pb", "Ag", "Bi", "S", "synthesized"]:
        if w in model.wv:
            print(w, [x for x, _ in model.wv.most_similar(w, topn=6)])
