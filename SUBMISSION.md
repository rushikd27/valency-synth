# OSI Hackathon submission kit (due Thu Oct 22, 7:00 PM CT)

## Checklist (on Moment: one member submits for the team)
- [ ] Every team member signs up on Moment (Globus preferred, Auth0 fallback) and joins the same team
- [ ] Video, 4 min max, at normal speaking speed
- [ ] Public post with the video (YouTube unlisted/public, X, LinkedIn, Bluesky, ...) that doesn't need a login to view
- [ ] One-page write-up: `WRITEUP.pdf`
- [ ] Code: https://github.com/rushikd27/valency-synth
- [ ] Platforms used: tick **Valency** (required for the Valency sponsor prize); Claude Code if listed

## Video script (~3:30 at normal pace)
Record a screen share with voice-over (Zoom, QuickTime, OBS, or Loom).

**0:00–0:25 | Hook** (show WRITEUP.pdf title)
"DFT predicts thousands of halide and chalcogenide perovskites, but only a few hundred have ever been made, and a score alone doesn't tell a chemist how to make one. We built Valency-Synth: a synthesizability model that checks itself against the literature and returns synthesis routes for its top candidates."

**0:25–1:00 | Corpus** (show figure panel a and `data/valency/` in the repo)
"With Valency we harvested every paper with 'perovskite' in the title across PubMed, arXiv and ChemRxiv: about 29,000 abstracts with dates. A formula parser pulls out 12,000 mentions of 1,900 non-oxide compositions and tags whether each abstract reports synthesis or only theory."

**1:00–1:35 | Model** (show README pipeline or `src/` list)
"The core is our positive-unlabeled bagging model from our MLST paper: composition descriptors like tolerance factors, plus crystal structure and DFT decomposition energy and band gap. On top, we added Valency layers: word2vec embeddings learned from the abstracts, and similarity to the closest compound the literature reports as synthesized. Literature counts are never model inputs, so the labels can't leak."

**1:35–2:30 | Results** (show figure panels b and c)
"The key test doesn't depend on our own labels. 23 compounds are unlabeled in our dataset, but the literature says they've been made: Cs2AgSbCl6, Cs2SnBr6, MASnBr3. The model never saw them as positives, yet it ranks them at the 88th percentile, and the 91st with the Valency neighbour layer. For a prospective test we rebuilt the layers from pre-2020 papers only: compounds first made after 2020 are recovered 77% of the time, versus 62% without the layer. One honest negative result: word2vec didn't help. The corpus also flagged label errors; for example, Cs2AgBiBr6, with 154 papers, was marked theoretical."

**2:30–3:15 | Recipes** (scroll `results/candidate_routes.md`)
"For each top candidate we pull synthesis routes: directly, if the compound has been reported, or by analogy to its nearest synthesized neighbours. Cs2AgGaCl6, for instance, points to room-temperature precipitation or hot-injection used for the Cs2AgInCl6 family, with links to the papers."

**3:15–3:35 | Close**
"Literature, then a check on our labels, then features, candidates and recipes: a ranked, referenced to-do list for a synthesis lab that updates as Valency ingests new papers. Code is open on GitHub."

## Post text
> Valency-Synth: which DFT-predicted perovskites can actually be made, and how? We mined 28,839 perovskite abstracts with @Valency, validated our PU synthesizability model on literature-confirmed compounds it never saw (88th–91st percentile), and attached literature synthesis routes to the top candidates. #OSIHackathon
> Video: <link>  Code: https://github.com/rushikd27/valency-synth
