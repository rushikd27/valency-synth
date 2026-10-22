"""Parse perovskite names into site-resolved compositions and compute descriptors.

Molecular A-site cations (MA, FA, GA) are kept as pseudo-species with effective
radii from Kieslich, Sun & Cheetham, Chem. Sci. 2014 (MA+ 2.17, FA+ 2.53, GA+ 2.78 A).
Sn2+ (absent from Shannon's table) uses 1.10 A from Travis et al., Chem. Sci. 2016.
"""
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np
from pymatgen.core import Composition, Element, Species

HALOGENS = {"F": -1, "Cl": -1, "Br": -1, "I": -1}
CHALCOGENS = {"O": -2, "S": -2, "Se": -2, "Te": -2}
ANIONS = {**HALOGENS, **CHALCOGENS}
MOLECULAR = {"MA": {"radius": 2.17, "charge": 1}, "FA": {"radius": 2.53, "charge": 1}, "GA": {"radius": 2.78, "charge": 1}}
RADIUS_OVERRIDES = {("Sn", 2): 1.10}
A_CANDIDATES = {"K", "Rb", "Cs", "Na", "Li", "Ca", "Sr", "Ba", "MA", "FA", "GA"}

_ALIASES = [("(CH3NH3)", "MA"), ("CH3NH3", "MA"), ("HC(NH2)2", "FA"), ("CH(NH2)2", "FA")]


def parse_name(name):
    """Return a Counter of species -> amount, expanding parentheses.

    MA/FA are matched before element symbols so FA is never read as F.
    """
    s = name.strip()
    for a, b in _ALIASES:
        s = s.replace(a, b)

    def expand(text):
        # innermost parentheses first: (AgSb)0.0625 -> Ag0.0625Sb0.0625
        pat = re.compile(r"\(([^()]*)\)(\d*\.?\d*)")
        while "(" in text:
            def rep(m):
                mult = float(m.group(2)) if m.group(2) not in ("", ".") else 1.0
                inner = _tokens(m.group(1))
                return "".join(f"{k}{v * mult:g}" for k, v in inner.items())
            new = pat.sub(rep, text)
            if new == text:
                break
            text = new
        return text

    return _tokens(expand(s))


def _tokens(text):
    c = Counter()
    for sym, num in re.findall(r"(MA|FA|GA|[A-Z][a-z]?)(\d*\.?\d*)", text):
        c[sym] += float(num) if num not in ("", ".") else 1.0
    return c


def normalize_abx3(c):
    """Scale so the anion total is 3 (ABX3 basis)."""
    x = sum(v for k, v in c.items() if k in ANIONS)
    if x <= 0:
        return None
    f = 3.0 / x
    return {k: v * f for k, v in c.items()}


def _shannon(sym, ox, cn_pref):
    if (sym, ox) in RADIUS_OVERRIDES:
        return RADIUS_OVERRIDES[(sym, ox)]
    r = _shannon_raw(sym, ox, cn_pref)
    if r is None:  # e.g. mixed-valence Au2+: average the neighbouring states
        lo, hi = _shannon_raw(sym, ox - 1, cn_pref), _shannon_raw(sym, ox + 1, cn_pref)
        if lo is not None and hi is not None:
            r = (lo + hi) / 2
    return r


def _shannon_raw(sym, ox, cn_pref):
    try:
        sp = Species(sym, ox)
    except Exception:
        return None
    for cn in cn_pref:
        try:
            return float(sp.get_shannon_radius(cn))
        except Exception:
            continue
    try:
        return float(sp.ionic_radius) if sp.ionic_radius is not None else None
    except Exception:
        return None


@lru_cache(maxsize=None)
def _elem(sym):
    return Element(sym)


def assign_sites(norm):
    """Split species into A, B, X sites on the ABX3 basis.

    A-site is filled (to a total of 1) by the largest A-type cations; any
    remaining A-type cation (e.g. K in Cs2KYCl6) goes to the B site.
    """
    X = {k: v for k, v in norm.items() if k in ANIONS}
    cations = {k: v for k, v in norm.items() if k not in ANIONS}

    def size(k):
        if k in MOLECULAR:
            return MOLECULAR[k]["radius"]
        r = _shannon(k, 1 if _elem(k).group == 1 else 2, ["XII", "VIII", "VI"])
        return r or 0.0

    A, B = {}, {}
    room = 1.0
    for k in sorted((k for k in cations if k in A_CANDIDATES), key=size, reverse=True):
        take = min(cations[k], room)
        if take > 1e-9:
            A[k] = take
            room -= take
        if cations[k] - take > 1e-9:
            if k in MOLECULAR:  # molecular cations never sit on B; allow A > 1
                A[k] = A.get(k, 0) + cations[k] - take
            else:
                B[k] = cations[k] - take
    for k, v in cations.items():
        if k not in A_CANDIDATES:
            B[k] = B.get(k, 0) + v
    return A, B, X


def _cation_charge(sym):
    if sym in MOLECULAR:
        return MOLECULAR[sym]["charge"]
    g = _elem(sym).group
    return 1 if g == 1 else 2 if g == 2 else None


def _b_oxidation_states(B, needed_total):
    """Pick common oxidation states for B-site species whose weighted sum
    matches the charge needed for neutrality (falls back to most common)."""
    options, common = {}, {}
    for k in B:
        e = _elem(k)
        common[k] = {s for s in (e.common_oxidation_states or ()) if s > 0}
        options[k] = sorted({s for s in e.oxidation_states if 0 < s <= 6} | common[k]) or [2]
    best, best_score, best_err = None, math.inf, math.inf

    def rec(keys, acc, total, penalty):
        nonlocal best, best_score, best_err
        if not keys:
            err = abs(total - needed_total)
            # charge balance dominates; uncommon states break ties
            score = err + 0.01 * penalty
            if score < best_score:
                best, best_score, best_err = dict(acc), score, err
            return
        k = keys[0]
        for s in options[k]:
            acc[k] = s
            rec(keys[1:], acc, total + s * B[k], penalty + (s not in common[k]))

    rec(list(B), {}, 0.0, 0)
    return best or {}, best_err


def _wmean(d, f):
    vals = [(v, f(k)) for k, v in d.items()]
    vals = [(w, x) for w, x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    tot = sum(w for w, _ in vals)
    return sum(w * x for w, x in vals) / tot if tot > 0 else np.nan


def _entropy(d):
    tot = sum(d.values())
    if tot <= 0:
        return 0.0
    p = [v / tot for v in d.values() if v > 0]
    return -sum(x * math.log(x) for x in p)


def describe(name):
    """Descriptor dict for one compound name (NaN where undefined)."""
    raw = parse_name(name)
    norm = normalize_abx3(raw)
    out = {"parse_ok": 0}
    if not norm:
        return out
    for k in norm:
        if k not in MOLECULAR:
            try:
                _elem(k)
            except ValueError:
                return out  # malformed name (e.g. stray 'G', 'E' tokens)
    A, B, X = assign_sites(norm)
    if not A or not B or not X:
        return out
    out["parse_ok"] = 1

    qA = {k: _cation_charge(k) or 1 for k in A}
    qX = {k: ANIONS[k] for k in X}
    charge_A = sum(qA[k] * v for k, v in A.items())
    charge_X = sum(qX[k] * v for k, v in X.items())
    oxB, charge_err = _b_oxidation_states(B, -(charge_A + charge_X))

    rA = _wmean(A, lambda k: MOLECULAR[k]["radius"] if k in MOLECULAR else _shannon(k, qA[k], ["XII", "VIII", "VI"]))
    rB = _wmean(B, lambda k: _shannon(k, oxB.get(k, 2), ["VI", "IV", "VIII"]))
    rX = _wmean(X, lambda k: _shannon(k, qX[k], ["VI"]))
    nA = _wmean(A, lambda k: qA[k])

    def chi(k):
        if k in MOLECULAR:
            return np.nan
        x = _elem(k).X
        return float(x) if x and not math.isnan(x) else np.nan

    out.update({
        "rA": rA, "rB": rB, "rX": rX,
        "tolerance_t": (rA + rX) / (math.sqrt(2) * (rB + rX)) if all(map(np.isfinite, [rA, rB, rX])) else np.nan,
        "octahedral_mu": rB / rX if np.isfinite(rB) and np.isfinite(rX) and rX else np.nan,
        "chiA": _wmean(A, chi), "chiB": _wmean(B, chi), "chiX": _wmean(X, chi),
        "oxB_mean": _wmean(B, lambda k: oxB.get(k)),
        "charge_balance_err": charge_err,
        "B_total": sum(B.values()),            # <1 for vacancy-ordered (A2BX6, A3B2X9)
        "A_entropy": _entropy(A), "B_entropy": _entropy(B), "X_entropy": _entropy(X),
        "n_species": len(norm),
        "organic_A_frac": sum(v for k, v in A.items() if k in MOLECULAR),
        "is_halide": float(all(k in HALOGENS for k in X)),
        "is_chalcogenide": float(all(k in CHALCOGENS for k in X)),
        "B_distinct": len(B),
    })
    # Bartel et al. (Sci. Adv. 2019) tau = rX/rB - nA (nA - (rA/rB)/ln(rA/rB))
    if all(map(np.isfinite, [rA, rB, rX, nA])) and rA > rB > 0:
        ratio = rA / rB
        out["bartel_tau"] = rX / rB - nA * (nA - ratio / math.log(ratio))
    else:
        out["bartel_tau"] = np.nan
    out["chi_diff_XB"] = out["chiX"] - out["chiB"]

    # composition-weighted elemental statistics over inorganic species
    inorg = {k: v for k, v in norm.items() if k not in MOLECULAR}
    props = {
        "Z": lambda e: e.Z, "mass": lambda e: float(e.atomic_mass), "row": lambda e: e.row,
        "group": lambda e: e.group, "X": lambda e: e.X, "atomic_radius": lambda e: float(e.atomic_radius) if e.atomic_radius else np.nan,
        "melting_point": lambda e: float(e.melting_point) if e.melting_point else np.nan,
    }
    w = np.array(list(inorg.values()), float)
    for p, f in props.items():
        vals = []
        for k in inorg:
            try:
                vals.append(float(f(_elem(k))))
            except Exception:
                vals.append(np.nan)
        v = np.array(vals, float)
        m = np.isfinite(v)
        if m.any():
            ww = w[m] / w[m].sum()
            mean = float((ww * v[m]).sum())
            out[f"{p}_mean"] = mean
            out[f"{p}_std"] = float(np.sqrt((ww * (v[m] - mean) ** 2).sum()))
            out[f"{p}_range"] = float(v[m].max() - v[m].min())
    return out


def reduced_key(name):
    """Canonical composition key for matching across sources (MA/FA kept as tokens)."""
    raw = parse_name(name)
    norm = normalize_abx3(raw) or raw
    tot = sum(norm.values())
    return tuple(sorted((k, round(v / tot, 3)) for k, v in norm.items()))
