"""Step 8: build a blind labelling tool from a FRESH random sample of the ML-ready articles.

The sample never contains the 3,000 development articles (seed 42) that the rules were tuned on.
Writes (all local, they contain article text):
  analysis_output/review_sample.csv   the 200 articles (used later by the scoring step)
  analysis_output/review_tool.html    open this in a browser and label the articles
The tool does NOT show the rule-based prediction, so your labels are independent.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys

import config
import text_profile

SHOW_CHARS = 1500

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind labelling tool</title>
<style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f6f6f4;color:#1c1c1c}
main{max-width:820px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:18px;margin:0 0 4px}.sub{color:#555;font-size:13px;margin-bottom:14px}
.bar{height:8px;background:#ddd;border-radius:4px;overflow:hidden;margin:6px 0 4px}.bar i{display:block;height:100%;background:#2f6fb5}
.card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:18px;margin:14px 0}
#title{font-size:20px;font-weight:600;margin:0 0 10px}#lead{white-space:pre-wrap;line-height:1.5;font-size:15px}
.btns{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin:12px 0}
button{font:inherit;padding:10px 12px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer;text-align:left}
button:hover{border-color:#2f6fb5}button.on{background:#2f6fb5;color:#fff;border-color:#2f6fb5}
.nav button{text-align:center}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
details{background:#fff;border:1px solid #ddd;border-radius:10px;padding:10px 14px;font-size:14px;line-height:1.5}
kbd{background:#eee;border:1px solid #ccc;border-radius:4px;padding:0 5px;font-size:12px}
.muted{color:#666;font-size:13px}
</style></head><body><main>
<h1>Blind labelling tool</h1>
<div class="sub">Label each article by its <b>main subject</b>. The tool does not show any computer prediction.
Progress is saved in this browser; export your labels at any time.</div>
<details><summary><b>Category definitions (click to open)</b></summary>
<p><b>market_stocks</b>: stock markets, shares, company earnings, investments, banking, economic indicators.<br>
<b>technology</b>: AI, software, cybersecurity, robotics, semiconductors, tech companies and gadgets, crypto/blockchain.<br>
<b>health</b>: medical research, diseases, medicines, hospitals, healthcare systems, public health.<br>
<b>energy</b>: oil, gas, electricity, renewables, solar, wind, nuclear, energy companies and markets.<br>
<b>politics</b>: elections, governments, parties, parliament, laws, diplomacy, war and international affairs, government policy.<br>
<b>other</b>: enough text, but about something else (sports, entertainment, crime, lifestyle, science, local news...).<br>
<b>uncertain</b>: too little text, unreadable, or two topics are genuinely equal.<br>
<b>not_news</b>: not an article (company profile page, shop or coupon page, list of links).<br>
Choose the <b>primary</b> subject, not just a keyword. An AI story about a stock price is market_stocks if the focus is the price.</p></details>
<div class="bar"><i id="fill"></i></div><div class="muted" id="prog"></div>
<div class="card"><h2 id="title"></h2><div id="lead"></div>
<div class="muted" id="more" style="margin-top:8px"></div></div>
<div class="btns" id="btns"></div>
<div class="row nav"><button id="back">&larr; Back</button><button id="next">Next &rarr;</button>
<button id="export"><b>Export my labels (CSV)</b></button><button id="reset">Reset</button></div>
<p class="muted">Keys: <kbd>1</kbd>-<kbd>8</kbd> label and move on, <kbd>&larr;</kbd> <kbd>&rarr;</kbd> move.</p>
</main>
<script>
const ITEMS = __DATA__;
const RUN = "__RUN__";
const LABELS = ["market_stocks","technology","health","energy","politics","other","uncertain","not_news"];
const KEY = "label_tool_" + RUN;
let state = {labels:{}, pos:0}, expanded = false;
try { const s = JSON.parse(localStorage.getItem(KEY)); if (s && s.labels) state = s; } catch (e) {}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
const $ = id => document.getElementById(id);
function render(){
  const it = ITEMS[state.pos];
  $("title").textContent = it.title;
  const long = it.lead.length > 700;
  $("lead").textContent = (long && !expanded) ? it.lead.slice(0,700) + " ..." : it.lead;
  const more = $("more"); more.textContent = "";
  if (long) { const a = document.createElement("a"); a.href = "#"; a.textContent = expanded ? "show less" : "show more";
    a.onclick = e => { e.preventDefault(); expanded = !expanded; render(); }; more.appendChild(a); }
  const done = Object.keys(state.labels).length;
  $("prog").textContent = "Article " + (state.pos+1) + " of " + ITEMS.length + "  |  labelled: " + done;
  $("fill").style.width = (100*done/ITEMS.length) + "%";
  const box = $("btns"); box.textContent = "";
  LABELS.forEach((l, i) => { const b = document.createElement("button"); b.textContent = (i+1) + "  " + l;
    if (state.labels[it.rid] === l) b.className = "on"; b.onclick = () => choose(i); box.appendChild(b); });
}
function go(d){ state.pos = Math.max(0, Math.min(ITEMS.length-1, state.pos + d)); expanded = false; save(); render(); }
function choose(i){ state.labels[ITEMS[state.pos].rid] = LABELS[i]; save();
  if (state.pos < ITEMS.length-1) { go(1); } else { render(); } }
function csvCell(v){ return '"' + String(v).replace(/"/g,'""') + '"'; }
function exportCsv(){
  const rows = ["review_id,article_id,your_label"];
  ITEMS.forEach(it => { if (state.labels[it.rid]) rows.push([it.rid, it.id, state.labels[it.rid]].map(csvCell).join(",")); });
  const blob = new Blob([rows.join("\\n") + "\\n"], {type:"text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "my_labels__TAG__.csv";
  document.body.appendChild(a); a.click(); a.remove();
}
$("back").onclick = () => go(-1); $("next").onclick = () => go(1); $("export").onclick = exportCsv;
$("reset").onclick = () => { if (confirm("Delete all your labels in this browser?")) { state = {labels:{}, pos:0}; save(); render(); } };
document.addEventListener("keydown", e => {
  if (e.key >= "1" && e.key <= "8") choose(Number(e.key)-1);
  else if (e.key === "ArrowLeft") go(-1); else if (e.key === "ArrowRight") go(1);
});
render();
</script></body></html>
"""


def safe_json(obj):
    """JSON that is safe inside a <script> block (no '</script>' or line-separator breakouts)."""
    return (json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def build_html(items, tag=""):
    run = hashlib.sha256("|".join(i["id"] for i in items).encode()).hexdigest()[:12]
    return (PAGE.replace("__DATA__", safe_json(items)).replace("__RUN__", run)
            .replace("__TAG__", ("_" + tag) if tag else ""))


def make_items(sample, seed):
    rows = [{"id": str(i), "title": str(t), "lead": str(b)[:SHOW_CHARS]}
            for i, t, b in zip(sample["id"], sample["title"], sample["body"])]
    random.Random(seed).shuffle(rows)               # random order: stopping early is still unbiased
    for n, r in enumerate(rows, 1):
        r["rid"] = n
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the blind labelling tool (fresh sample).")
    ap.add_argument("--n", type=int, default=config.REVIEW_N)
    ap.add_argument("--seed", type=int, default=config.REVIEW_SEED)
    ap.add_argument("--tag", default="", help="e.g. round2: new files, the first test files stay untouched")
    ap.add_argument("--exclude-sample", action="append", default=[],
                    help="a review_sample.csv whose articles must not be drawn again (repeatable)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = os.path.join(config.VALIDATION_DIR, config.ML_READY_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"Not found: {path}")
    print("Step 1/2: finding the 3,000 development articles to exclude (about 1 minute)...")
    dev, total = text_profile.sample_articles(path, config.SAMPLE_SIZE, config.SAMPLE_SEED,
                                              config.CHUNK_ROWS, config.LEAD_CHARS)
    dev_ids = set(dev["id"])
    for ex in args.exclude_sample:
        if not os.path.exists(ex):
            raise SystemExit(f"Not found: {ex}")
        with open(ex, newline="", encoding="utf-8-sig") as fh:
            dev_ids |= {row["id"] for row in csv.DictReader(fh)}
    print(f"Step 2/2: drawing {args.n} fresh articles from the other {total - len(dev_ids):,} (about 1 minute)...")
    fresh, _ = text_profile.sample_articles(path, args.n, args.seed, config.CHUNK_ROWS,
                                            config.LEAD_CHARS, exclude_ids=dev_ids)
    assert not (set(fresh["id"]) & dev_ids), "fresh sample overlaps the development sample"
    items = make_items(fresh, args.seed)
    suffix = f"_{args.tag}" if args.tag else ""
    fresh_domains = fresh["url"].map(text_profile.domain_of)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, f"review_sample{suffix}.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["review_id", "id", "domain", "title", "lead"])
        dom = dict(zip(fresh["id"].astype(str), fresh_domains))
        for it in items:
            w.writerow([it["rid"], it["id"], dom.get(it["id"], ""), it["title"], it["lead"]])
    out = os.path.join(config.OUTPUT_DIR, f"review_tool{suffix}.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(build_html(items, args.tag))
    print(f"\nDone. {len(items)} articles, none from the development sample.")
    print(f"Open this file in your browser (double-click it):\n  {out}")


if __name__ == "__main__":
    sys.exit(main())
