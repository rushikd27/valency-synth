# valency-synth

Bagged positive-unlabeled (PU) synthesizability model for halide and chalcogenide
perovskites, extending Desai et al., arXiv:2510.06166, with literature features
mined from the Valency research corpus.

## Layout
- `data/raw/final_data.csv` — labeled dataset (`theoretical=False` = synthesized), DFT decomposition energy and band gap
- `src/composition.py` — parses compound names (MA/FA/GA, parentheses, double/vacancy-ordered) into A/B/X sites; radii, tolerance/octahedral/Bartel-tau factors, electronegativity, mixing entropy, elemental statistics
- `src/pu.py` — bagged PU learning (Mordelet & Vert 2014) and hidden-positive cross-validation grouped by composition
- `src/features.py` — feature-set assembly
- `src/run_baseline.py` — baseline ablation (no literature features)
- `src/harvest_plan.py`, `src/collect_harvest.py` — date windows for harvesting every 'perovskite'-titled paper from Valency (<=200 results per call) and collection of the saved responses into `data/valency/raw/` (local only)
- `src/literature.py` — builds `data/valency/corpus.jsonl.gz` (28,839 papers: PubMed, arXiv, ChemRxiv) and `data/valency/mentions.csv` (formula mentions, dates, synthesis vs. theory context)
- `src/embeddings.py` — mat2vec-style word2vec; formulas split into species tokens, compositions embedded as fraction-weighted species vectors (works for unseen formulas)
- `src/lit_layers.py` — Valency feature layers (word2vec PCs + concept cosines; similarity to nearest literature-synthesized composition outside the dataset) and per-composition literature table
- `src/run_valency.py` — layer ablation, external test on literature-confirmed unlabeled compositions, pre-2020 time split
- `src/recipes.py` — synthesis routes for top candidates from direct mentions or nearest synthesized neighbours
- `src/check_leakage.py` — DFT/source confound checks
- `results/` — ablation outputs

## Run
```
pip install scikit-learn gensim && pip install --no-deps pymatgen-core
python3 src/run_baseline.py 100   # number of bags
python3 src/literature.py          # after harvesting (needs data/valency/raw)
python3 src/embeddings.py && python3 src/embeddings.py 2020-01-01
python3 src/run_valency.py 50
python3 src/recipes.py "all | base+nbr" 25
```

## Status
- [x] Baseline PU ablation (composition / +structure / +DFT)
- [x] Valency abstract corpus with first-report dates
- [x] Literature features: mentions, word2vec embeddings, synthesized-neighbor similarity
- [x] Ablations + temporal split
- [x] Recipe routes for top candidates (abstract level; full recipes via Scite where indexed)

## Results (50 bags; `results/valency_ablation.csv`)
Hidden-positive CV on the dataset labels, plus an external test: compositions
unlabeled in final_data.csv that the Valency corpus reports in a synthesis
context (never used as positives).

| layers built from | feature set | recall@0.5 | PU-AUC | ext AUC | ext median pctile | ext recall@0.5 |
|---|---|---|---|---|---|---|
| all abstracts (n_ext=23) | base (comp+struct+DFT) | 0.858 | 0.916 | 0.858 | 0.882 | 0.478 |
| | base+w2v | 0.837 | 0.907 | 0.850 | 0.889 | 0.565 |
| | base+nbr | 0.846 | 0.911 | 0.864 | 0.908 | 0.565 |
| | base+w2v+nbr | 0.825 | 0.907 | 0.861 | 0.897 | 0.565 |
| pre-2020 abstracts, test = first reported 2020+ (n_ext=13) | base | 0.858 | 0.916 | 0.904 | 0.951 | 0.615 |
| | base+w2v | 0.829 | 0.906 | 0.880 | 0.948 | 0.615 |
| | base+nbr | 0.874 | 0.913 | 0.918 | 0.953 | 0.769 |
| | base+w2v+nbr | 0.833 | 0.908 | 0.904 | 0.948 | 0.692 |

Candidate routes for the top 25 unlabeled compositions (model `all | base+nbr`):
`results/candidate_routes.md`.
