# valency-synth

Bagged positive-unlabeled (PU) synthesizability model for halide and chalcogenide
perovskites, extending Desai et al., arXiv:2510.06166, with literature features
mined from the Valency research corpus.

## Layout
- `data/raw/final_data.csv` — labeled dataset (`theoretical=False` = synthesized), DFT decomposition energy and band gap
- `src/composition.py` — parses compound names (MA/FA/GA, parentheses, double/vacancy-ordered) into A/B/X sites; radii, tolerance/octahedral/Bartel-tau factors, electronegativity, mixing entropy, elemental statistics
- `src/pu.py` — bagged PU learning (Mordelet & Vert 2014) and hidden-positive cross-validation grouped by composition
- `src/features.py` — feature-set assembly
- `src/run_baseline.py` — baseline ablation (no literature features yet)
- `results/` — ablation outputs

## Run
```
pip install scikit-learn gensim && pip install --no-deps pymatgen-core
python3 src/run_baseline.py 100   # number of bags
```

## Status
- [x] Baseline PU ablation (composition / +structure / +DFT)
- [ ] Valency abstract corpus with first-report dates
- [ ] Literature features: mentions, word2vec embeddings, synthesized-neighbor similarity
- [ ] Ablations + temporal split
- [ ] Recipe recommendation for top candidates
