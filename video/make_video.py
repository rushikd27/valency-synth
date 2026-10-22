"""Build the hackathon video: HTML slides -> PNG (headless Chromium), narration
with Kokoro TTS (local), stitched with ffmpeg. Usage: python3 video/make_video.py <tts_dir>"""
import os
import subprocess
import sys

import imageio_ffmpeg
import pandas as pd
import soundfile as sf
from PIL import Image

TTS = sys.argv[1]
OUT = "video/build"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs(OUT, exist_ok=True)

# figure panels, exported individually by src/make_figure.py
for tag in "abc":
    Image.open(f"results/figure_{tag}.png").save(f"{OUT}/panel_{tag}.png")

routes = pd.read_csv("results/candidate_routes.csv").head(7)
rows = "".join(
    f"<tr><td class=f>{r.Name}</td><td>{r.score:.2f}</td><td>{r.evidence}</td>"
    f"<td>{'; '.join(x.split(' x')[0] for x in str(r.routes).split('; ')[:2]) if isinstance(r.routes, str) else '—'}</td></tr>"
    for r in routes.itertuples())

CSS = """
*{box-sizing:border-box} body{margin:0;width:1920px;height:1080px;background:#fbfaf7;color:#1f1f1e;
font-family:'Helvetica Neue',Arial,sans-serif;padding:90px 120px;display:flex;flex-direction:column}
h1{font-size:68px;margin:0 0 28px;letter-spacing:-1px} h2{font-size:54px;margin:0 0 36px;letter-spacing:-.5px}
p,li{font-size:36px;line-height:1.45;color:#3a3936} li{margin:10px 0}
.k{color:#1c5cab;font-weight:700} .o{color:#d95926;font-weight:700} .muted{color:#6b6a64}
.tag{font-size:26px;letter-spacing:3px;text-transform:uppercase;color:#6b6a64;margin-bottom:22px}
.row{display:flex;gap:70px;align-items:center;flex:1} .row img{height:760px} .grow{flex:1}
.big{font-size:120px;font-weight:800;color:#1c5cab;line-height:1} .stat{margin-bottom:46px}
.stat div:last-child{font-size:32px;color:#6b6a64}
.flow{display:flex;align-items:center;gap:22px;margin-top:40px;flex-wrap:wrap}
.box{background:#fff;border:2px solid #d8d6cf;border-radius:16px;padding:26px 30px;font-size:31px;line-height:1.3;width:310px;min-height:190px}
.box b{display:block;font-size:34px;margin-bottom:8px;color:#1c5cab} .arr{font-size:50px;color:#9a988f}
table{border-collapse:collapse;font-size:31px;width:100%} td,th{padding:16px 20px;border-bottom:1px solid #e4e3de;text-align:left}
th{color:#6b6a64;font-weight:600;font-size:26px;text-transform:uppercase;letter-spacing:1px} td.f{font-weight:700}
"""

SLIDES = [
    ("""<div class=tag>Open Scientific Intelligence Hackathon · 2026</div>
<div style="flex:1;display:flex;flex-direction:column;justify-content:center">
<h1 style="font-size:104px;margin-bottom:34px">Valency-Synth</h1>
<p style="font-size:44px;max-width:1500px">Which predicted perovskites can actually be made, and <span class=k>how</span>?
Literature-grounded synthesizability and synthesis routes for non-oxide perovskites.</p>
<p class=muted style="margin-top:60px">Rushik Desai · Juan Carlos Verduzco · Purdue University</p></div>""",
     "Density functional theory predicts thousands of halide and chalcogenide perovskites. "
     "But only a few hundred have ever been made, and a score alone does not tell a chemist how to make one. "
     "We built Valency Synth: a synthesizability model that checks itself against the literature, "
     "and returns synthesis routes for its top candidates."),
    ("""<div class=tag>The gap</div><h2>A synthesizability score is not a synthesis plan</h2>
<ul><li>Our PU-learning model (Desai et al., MLST) learns from <span class=k>1,228</span> compositions and polymorphs, only <span class=k>251</span> synthesized</li>
<li>Inputs: composition descriptors, crystal structure, DFT decomposition energy and band gap</li>
<li><b>Open question 1:</b> can we validate it <i>without</i> trusting our own labels?</li>
<li><b>Open question 2:</b> for a high-scoring candidate, <i>how</i> would you make it?</li></ul>""",
     "Our starting point is the positive unlabeled learning model from our published work. "
     "It learns from about twelve hundred compositions and polymorphs, only two hundred and fifty one of them synthesized, "
     "using composition, crystal structure, and D F T stability. "
     "Two questions were left open. Can we validate it without trusting our own labels? "
     "And once a candidate scores high, how would you actually make it?"),
    ("""<div class=tag>Step 1 · Valency</div><h2>Every perovskite paper, with dates</h2>
<div class=row><img src="panel_a.png"><div class=grow>
<div class=stat><div class=big>28,839</div><div>abstracts · PubMed, arXiv, ChemRxiv</div></div>
<div class=stat><div class=big>11,929</div><div>formula mentions · 1,914 non-oxide compositions</div></div>
<p style="font-size:30px">Each mention tagged <span class=k>synthesis</span> vs <span class=o>theory</span> context, with first-report date</p></div></div>""",
     "Step one is the literature. Using Valency, we harvested every paper with perovskite in the title, "
     "across PubMed, arXiv and ChemRxiv: almost twenty nine thousand abstracts, each with a date. "
     "A formula parser then extracts nearly twelve thousand mentions of about nineteen hundred non oxide compositions, "
     "and tags whether each abstract reports a synthesis, or only a calculation."),
    ("""<div class=tag>Step 2 · Model</div><h2>Bagged PU learning + Valency feature layers</h2>
<div class=flow>
<div class=box><b>Composition + DFT</b>tolerance &amp; octahedral factors, radii, E<sub>decomp</sub>, band gap</div><div class=arr>+</div>
<div class=box><b>word2vec</b>species vectors learned from 210k abstract sentences</div><div class=arr>+</div>
<div class=box><b>Neighbors</b>similarity to nearest literature-synthesized compound</div><div class=arr>→</div>
<div class=box style="border-color:#2a78d6"><b>Bagged PU</b>100 bags, out-of-bag scores, CV by composition</div></div>
<p style="margin-top:56px">Literature mention counts are <b>never</b> model inputs, so labels cannot leak.</p>""",
     "Step two is the model: bagged positive unlabeled learning, where every compound is scored only by bags that never saw it. "
     "On top of the composition and D F T features, we added two Valency layers. "
     "First, word to vec embeddings learned from the abstracts. "
     "Second, similarity to the closest compound the literature reports as synthesized. "
     "Importantly, literature counts are never model inputs, so the labels cannot leak."),
    ("""<div class=tag>Result 1 · label-independent validation</div><h2>Hidden positives the model never saw</h2>
<div class=row><img src="panel_b.png"><div class=grow>
<p>23 compounds are <b>unlabeled</b> in our data but reported as <b>synthesized</b> in the corpus: Cs<sub>2</sub>AgSbCl<sub>6</sub>, Cs<sub>2</sub>SnBr<sub>6</sub>, Cs<sub>2</sub>TiBr<sub>6</sub>, MASnBr<sub>3</sub>…</p>
<p>Ranked at the <span class=k>88th</span> percentile of unlabeled compounds, <span class=o>91st</span> with the neighbor layer.</p></div></div>""",
     "Here is the key test, and it does not depend on our own labels. "
     "Twenty three compounds are unlabeled in our dataset, but the literature says they have been made: "
     "for example cesium silver antimony chloride, cesium tin bromide, and methylammonium tin bromide. "
     "The model never saw them as positives. Yet it ranks them at the eighty eighth percentile of unlabeled compounds, "
     "and the ninety first with the Valency neighbor layer."),
    ("""<div class=tag>Result 2 · prospective test</div><h2>Built from pre-2020 papers, tested on 2020+ discoveries</h2>
<div class=row><img src="panel_c.png"><div class=grow>
<p>Recall on compounds first reported 2020 or later: <span class=k>0.62</span> → <span class=o>0.77</span> with the neighbor layer</p>
<p><b>Honest negative:</b> word2vec did not help</p>
<p><b>Bonus:</b> the corpus flagged label noise, e.g. Cs<sub>2</sub>AgBiBr<sub>6</sub> (154 papers) marked theoretical</p></div></div>""",
     "Next, a prospective test. We rebuilt the Valency layers from papers before twenty twenty, "
     "and tested on compounds first reported in twenty twenty or later. "
     "With the neighbor layer, recall rises from sixty two to seventy seven percent, on thirteen compounds. "
     "One honest negative result: the word to vec layer did not help. "
     "And the corpus exposed label errors in our own data: cesium silver bismuth bromide, with one hundred and fifty four papers, was marked theoretical."),
    (f"""<div class=tag>Step 3 · routes</div><h2>Top candidates, with literature synthesis routes</h2>
<table><tr><th>Candidate</th><th>Score</th><th>Evidence</th><th>Suggested routes</th></tr>{rows}</table>
<p style="font-size:28px;margin-top:30px" class=muted>Direct: the compound's own synthesis papers. Analogy: three nearest synthesized neighbors. Every route links to its papers.</p>""",
     "Finally, step three turns scores into plans. For each top candidate, we pull synthesis routes from the corpus: "
     "directly, if the compound has been reported, or by analogy to its three nearest synthesized neighbors. "
     "For instance, cesium silver gallium chloride points to room temperature precipitation and hot injection, "
     "as used for the cesium silver indium chloride family, with links to every source paper."),
    ("""<div style="flex:1;display:flex;flex-direction:column;justify-content:center">
<h1 style="font-size:80px">Literature → label check → features → candidates → recipes</h1>
<p style="font-size:40px;margin-top:20px">A ranked, referenced to-do list for a synthesis lab that updates as Valency ingests new papers.</p>
<p style="font-size:38px;margin-top:70px"><span class=k>github.com/rushikd27/valency-synth</span></p>
<p class=muted>Next: literature-confirmed positives · full experimental sections into structured recipes · oxides</p></div>""",
     "So the full loop is: literature, a check on our labels, features, candidates, and recipes. "
     "The result is a ranked and referenced to do list for a synthesis lab, which keeps updating as Valency ingests new papers. "
     "The code is open on GitHub. Thank you."),
]


def main():
    from kokoro_onnx import Kokoro
    k = Kokoro(f"{TTS}/kokoro.onnx", f"{TTS}/voices.bin")
    parts = []
    for i, (html, text) in enumerate(SLIDES):
        page = f"{OUT}/s{i}.html"
        open(page, "w").write(f"<!doctype html><meta charset=utf-8><style>{CSS}</style><body>{html}</body>")
        subprocess.run([CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                        "--window-size=1920,1080", f"--screenshot={OUT}/s{i}.png", os.path.abspath(page)],
                       check=True, capture_output=True)
        wav = f"{OUT}/s{i}.wav"
        if os.path.exists(wav) and open(f"{OUT}/s{i}.txt").read() == text if os.path.exists(f"{OUT}/s{i}.txt") else False:
            audio, sr = sf.read(wav)   # narration unchanged: reuse
        else:
            audio, sr = k.create(text, voice="am_michael", speed=1.0, lang="en-us")
            sf.write(wav, audio, sr)
            open(f"{OUT}/s{i}.txt", "w").write(text)
        dur = len(audio) / sr + 0.6
        clip = f"{OUT}/s{i}.mp4"
        subprocess.run([FFMPEG, "-y", "-loop", "1", "-i", f"{OUT}/s{i}.png", "-i", f"{OUT}/s{i}.wav",
                        "-af", "adelay=400|400,apad", "-t", f"{dur:.2f}", "-c:v", "libx264", "-tune", "stillimage",
                        "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "44100", clip],
                       check=True, capture_output=True)
        parts.append(clip)
        print(f"slide {i}: {dur:.1f}s", flush=True)
    open(f"{OUT}/list.txt", "w").write("".join(f"file '{os.path.basename(p)}'\n" for p in parts))
    subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", f"{OUT}/list.txt", "-c", "copy",
                    "video/valency_synth.mp4"], check=True, capture_output=True)
    print("wrote video/valency_synth.mp4")


if __name__ == "__main__":
    main()
