#!/usr/bin/env python3
"""Build a dependency-free human review packet for Kimi Task D-R parsing."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import exam_schema


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def legacy_first_keyword_verdict(text: str) -> tuple[str | None, str]:
    """Reproduce the pre-repair first-keyword parser for audit comparison."""
    value = (text or "").strip()
    if not value:
        return None, "fail"
    lowered = value.lower()
    if lowered in exam_schema._SUCCESS_WORDS:
        return "Success", "strict"
    if lowered in exam_schema._FAIL_WORDS:
        return "Fail", "strict"
    if "unsuccess" in lowered or "un-fail" in lowered or "unfail" in lowered:
        return "Fail", "recoverable"

    matches = [
        (match.start(), match.end(), match.group(1).lower())
        for match in exam_schema._VERDICT_RE.finditer(lowered)
    ]
    if not matches:
        return None, "fail"
    has_success = any(word in exam_schema._SUCCESS_WORDS for _, _, word in matches)
    has_fail = any(word in exam_schema._FAIL_WORDS for _, _, word in matches)
    if has_success and has_fail:
        success_matches = sorted(
            [item for item in matches if item[2] in exam_schema._SUCCESS_WORDS],
            key=lambda item: len(item[2]),
            reverse=True,
        )
        fail_matches = sorted(
            [item for item in matches if item[2] in exam_schema._FAIL_WORDS],
            key=lambda item: len(item[2]),
            reverse=True,
        )
        for start_s, end_s, _ in success_matches:
            for start_f, end_f, _ in fail_matches:
                if abs(start_s - start_f) <= 15 or abs(end_s - end_f) <= 15:
                    return "Fail", "recoverable"
        first_word = sorted(matches, key=lambda item: item[0])[0][2]
        return (
            "Success" if first_word in exam_schema._SUCCESS_WORDS else "Fail",
            "recoverable",
        )
    longest = max(matches, key=lambda item: len(item[2]))[2]
    if longest in exam_schema._SUCCESS_WORDS:
        return "Success", "recoverable"
    if longest in exam_schema._FAIL_WORDS:
        return "Fail", "recoverable"
    return None, "fail"


def build_records(score_path: Path, exam_path: Path) -> list[dict[str, Any]]:
    gold_by_uid = {
        row["uid"]: row.get("answer")
        for row in read_jsonl(exam_path)
    }
    records = []
    for row in read_jsonl(score_path):
        if row.get("parse_mode") == "strict":
            continue
        raw = row.get("raw", "")
        legacy_pred, legacy_mode = legacy_first_keyword_verdict(raw)
        corrected_pred, corrected_mode = exam_schema.parse_verdict(raw)
        records.append({
            "review_id": len(records) + 1,
            "uid": row["uid"],
            "env_task": row.get("env_task"),
            "group_id": row.get("group_id"),
            "variant": row.get("variant"),
            "visual": row.get("visual"),
            "framing": row.get("framing"),
            "gold": row.get("gold") or gold_by_uid.get(row["uid"]),
            "legacy_pred": legacy_pred,
            "legacy_mode": legacy_mode,
            "corrected_pred": corrected_pred,
            "corrected_mode": corrected_mode,
            "parser_changed": legacy_pred != corrected_pred,
            "raw_chars": len(raw),
            "raw_response": raw,
            "response_tail": raw[-600:],
            "human_verdict": "",
            "verdict_evidence": "",
            "confidence": "",
            "reviewer_id": "",
            "notes": "",
        })
    return sorted(
        records,
        key=lambda row: (
            not row["parser_changed"],
            row["corrected_mode"] != "fail",
            row["env_task"],
            row["variant"],
            row["uid"],
        ),
    )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def build_html(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    data = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    summary_json = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kimi Task D-R Parser Human Review</title>
<style>
:root {{ color-scheme: light; --ink:#16181d; --muted:#626873; --line:#d7dbe2;
  --bg:#f5f6f8; --panel:#fff; --accent:#1769aa; --warn:#a33a20; --ok:#237a45; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
header {{ position:sticky; top:0; z-index:3; background:var(--panel);
  border-bottom:1px solid var(--line); padding:12px 20px; }}
h1 {{ margin:0 0 8px; font-size:20px; letter-spacing:0; }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
button,select,input {{ min-height:34px; border:1px solid #aeb4bf; background:#fff;
  color:var(--ink); padding:6px 10px; border-radius:4px; font:inherit; }}
button {{ cursor:pointer; }}
button.primary {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
button:disabled {{ opacity:.45; cursor:not-allowed; }}
main {{ max-width:1180px; margin:0 auto; padding:18px 20px 48px; }}
.status {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); margin-bottom:14px; }}
.status div {{ background:var(--panel); padding:10px 12px; }}
.status strong {{ display:block; font-size:18px; }}
.meta {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); }}
.meta div {{ background:var(--panel); padding:9px 11px; min-width:0; }}
.meta span {{ display:block; color:var(--muted); font-size:12px; }}
.changed {{ color:var(--warn); font-weight:700; }}
.response {{ margin-top:14px; background:var(--panel); border:1px solid var(--line); }}
.response h2,.review h2 {{ margin:0; padding:10px 12px; font-size:15px;
  border-bottom:1px solid var(--line); }}
pre {{ margin:0; padding:14px; white-space:pre-wrap; overflow-wrap:anywhere;
  font:13px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; max-height:48vh; overflow:auto; }}
.review {{ margin-top:14px; background:var(--panel); border:1px solid var(--line); }}
.review-body {{ padding:14px; }}
fieldset {{ border:0; padding:0; margin:0 0 14px; }}
legend {{ font-weight:650; margin-bottom:7px; }}
.choices {{ display:flex; flex-wrap:wrap; gap:7px 14px; }}
label {{ cursor:pointer; }}
textarea {{ width:100%; min-height:76px; resize:vertical; border:1px solid #aeb4bf;
  border-radius:4px; padding:8px; font:inherit; }}
.gold {{ display:none; }}
body.show-gold .gold {{ display:block; }}
.hint {{ color:var(--muted); margin:7px 0 0; }}
.match {{ margin-top:8px; font-weight:650; }}
.match.ok {{ color:var(--ok); }} .match.bad {{ color:var(--warn); }}
@media (max-width:760px) {{
  .status,.meta {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  main {{ padding:12px 10px 36px; }} header {{ padding:10px; }}
}}
</style>
</head>
<body>
<header>
  <h1>Kimi Task D-R Parser Human Review</h1>
  <div class="toolbar">
    <label>Reviewer <input id="reviewer" placeholder="reviewer-id"></label>
    <select id="filter" aria-label="Review subset">
      <option value="changed">Priority: parser changed (49)</option>
      <option value="recoverable">All recoverable (132)</option>
      <option value="unparsable">Unparsable (1)</option>
      <option value="all">All review items (133)</option>
      <option value="pending">Pending in current subset</option>
    </select>
    <button id="prev" title="Previous item">Previous</button>
    <button id="next" title="Next item">Next</button>
    <button id="goldToggle">Show gold</button>
    <button id="exportCsv" class="primary">Export CSV</button>
    <button id="exportJson">Export JSON</button>
  </div>
</header>
<main>
  <section class="status">
    <div><span>Item</span><strong id="position">-</strong></div>
    <div><span>Completed</span><strong id="completed">0</strong></div>
    <div><span>Parser-changed reviewed</span><strong id="changedProgress">0 / 49</strong></div>
    <div><span>Stored locally</span><strong id="storageState">Yes</strong></div>
  </section>
  <section class="meta">
    <div><span>UID</span><b id="uid"></b></div>
    <div><span>Environment / variant</span><b id="condition"></b></div>
    <div><span>Presentation</span><b id="presentation"></b></div>
    <div><span>Parser change</span><b id="changed"></b></div>
    <div><span>Legacy parser</span><b id="legacy"></b></div>
    <div><span>Corrected parser</span><b id="corrected"></b></div>
    <div><span>Response length</span><b id="length"></b></div>
    <div class="gold"><span>Gold (do not use to infer intent)</span><b id="gold"></b></div>
  </section>
  <section class="response">
    <h2>Raw model response</h2>
    <pre id="raw"></pre>
  </section>
  <section class="review">
    <h2>Independent human judgment</h2>
    <div class="review-body">
      <fieldset>
        <legend>1. What verdict did the model ultimately intend?</legend>
        <div class="choices" id="verdictChoices"></div>
      </fieldset>
      <fieldset>
        <legend>2. What evidence supports that judgment?</legend>
        <div class="choices" id="evidenceChoices"></div>
      </fieldset>
      <fieldset>
        <legend>3. Confidence</legend>
        <div class="choices" id="confidenceChoices"></div>
      </fieldset>
      <label for="notes"><b>4. Notes</b></label>
      <textarea id="notes" placeholder="Quote the decisive ending or explain ambiguity."></textarea>
      <div id="match" class="match"></div>
      <p class="hint">Selections are stored in this browser. Export before switching machines or clearing browser data.</p>
    </div>
  </section>
</main>
<script>
const ITEMS = {data};
const SUMMARY = {summary_json};
const KEY = "gridwm-kimi-taskdr-parser-review-v1";
const verdictOptions = ["Success","Fail","Ambiguous","Unparsable"];
const evidenceOptions = [
  ["standalone_final","Standalone final verdict"],
  ["concluding_sentence","Unambiguous concluding sentence"],
  ["conflicting","Conflicting conclusions"],
  ["truncated","Truncated / no conclusion"]
];
const confidenceOptions = ["High","Medium","Low"];
let saved = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let currentIndex = 0;
let visible = [];
const $ = id => document.getElementById(id);
function escCsv(v) {{ const s=String(v ?? ""); return '"' + s.replaceAll('"','""') + '"'; }}
function answer(uid) {{ return saved[uid] || {{}}; }}
function isDone(uid) {{ const a=answer(uid); return !!(a.human_verdict && a.verdict_evidence && a.confidence && a.reviewer_id); }}
function persist() {{ localStorage.setItem(KEY, JSON.stringify(saved)); }}
function buildChoices(container, name, options) {{
  container.innerHTML = "";
  options.forEach(option => {{
    const value = Array.isArray(option) ? option[0] : option;
    const label = Array.isArray(option) ? option[1] : option;
    const node = document.createElement("label");
    node.innerHTML = `<input type="radio" name="${{name}}" value="${{value}}"> ${{label}}`;
    container.appendChild(node);
  }});
}}
function filtered() {{
  const mode=$("filter").value;
  let base=ITEMS;
  if(mode==="changed") base=ITEMS.filter(x=>x.parser_changed);
  if(mode==="recoverable") base=ITEMS.filter(x=>x.corrected_mode==="recoverable");
  if(mode==="unparsable") base=ITEMS.filter(x=>x.corrected_mode==="fail");
  if(mode==="pending") base=ITEMS.filter(x=>!isDone(x.uid));
  return base;
}}
function saveCurrent() {{
  if(!visible.length) return;
  const item=visible[currentIndex];
  const checked = name => document.querySelector(`input[name="${{name}}"]:checked`)?.value || "";
  saved[item.uid] = {{
    human_verdict: checked("verdict"),
    verdict_evidence: checked("evidence"),
    confidence: checked("confidence"),
    reviewer_id: $("reviewer").value.trim(),
    notes: $("notes").value.trim()
  }};
  persist();
}}
function render() {{
  visible=filtered();
  if(currentIndex>=visible.length) currentIndex=Math.max(0,visible.length-1);
  const item=visible[currentIndex];
  $("position").textContent=visible.length ? `${{currentIndex+1}} / ${{visible.length}}` : "0 / 0";
  $("completed").textContent=`${{ITEMS.filter(x=>isDone(x.uid)).length}} / ${{ITEMS.length}}`;
  const changed=ITEMS.filter(x=>x.parser_changed);
  $("changedProgress").textContent=`${{changed.filter(x=>isDone(x.uid)).length}} / ${{changed.length}}`;
  $("prev").disabled=!visible.length || currentIndex===0;
  $("next").disabled=!visible.length || currentIndex===visible.length-1;
  if(!item) return;
  $("uid").textContent=item.uid;
  $("condition").textContent=`${{item.env_task}} / ${{item.variant}}`;
  $("presentation").textContent=`${{item.visual}} / ${{item.framing}}`;
  $("changed").textContent=item.parser_changed ? "YES - priority" : "No";
  $("changed").className=item.parser_changed ? "changed" : "";
  $("legacy").textContent=item.legacy_pred || "Unparsable";
  $("corrected").textContent=item.corrected_pred || "Unparsable";
  $("length").textContent=`${{item.raw_chars}} chars`;
  $("gold").textContent=item.gold || "";
  $("raw").textContent=item.raw_response;
  const a=answer(item.uid);
  $("reviewer").value=a.reviewer_id || $("reviewer").value;
  for(const name of ["verdict","evidence","confidence"]) {{
    document.querySelectorAll(`input[name="${{name}}"]`).forEach(input => {{
      const key=name==="verdict"?"human_verdict":name==="evidence"?"verdict_evidence":"confidence";
      input.checked=input.value===(a[key] || "");
    }});
  }}
  $("notes").value=a.notes || "";
  const hv=a.human_verdict;
  const cp=item.corrected_pred || "Unparsable";
  $("match").textContent=hv ? (hv===cp ? "Corrected parser matches human verdict." : `Mismatch: human=${{hv}}, parser=${{cp}}`) : "";
  $("match").className="match " + (hv ? (hv===cp?"ok":"bad") : "");
}}
function move(delta) {{ saveCurrent(); currentIndex=Math.max(0,Math.min(visible.length-1,currentIndex+delta)); render(); window.scrollTo(0,0); }}
function exportRows(format) {{
  saveCurrent();
  const rows=ITEMS.map(item=>({{...item,...answer(item.uid),
    corrected_matches_human: answer(item.uid).human_verdict ?
      answer(item.uid).human_verdict===(item.corrected_pred || "Unparsable") : ""
  }}));
  let blob,name;
  if(format==="json") {{
    blob=new Blob([JSON.stringify({{summary:SUMMARY,reviews:rows}},null,2)],{{type:"application/json"}});
    name="kimi_taskdr_parser_review.json";
  }} else {{
    const fields=Object.keys(rows[0]);
    const text=[fields.map(escCsv).join(","),...rows.map(r=>fields.map(k=>escCsv(r[k])).join(","))].join("\\n");
    blob=new Blob([text],{{type:"text/csv;charset=utf-8"}});
    name="kimi_taskdr_parser_review.csv";
  }}
  const url=URL.createObjectURL(blob); const a=document.createElement("a");
  a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url);
}}
buildChoices($("verdictChoices"),"verdict",verdictOptions);
buildChoices($("evidenceChoices"),"evidence",evidenceOptions);
buildChoices($("confidenceChoices"),"confidence",confidenceOptions);
$("filter").addEventListener("change",()=>{{saveCurrent();currentIndex=0;render();}});
$("prev").addEventListener("click",()=>move(-1));
$("next").addEventListener("click",()=>move(1));
$("goldToggle").addEventListener("click",()=>{{
  document.body.classList.toggle("show-gold");
  $("goldToggle").textContent=document.body.classList.contains("show-gold")?"Hide gold":"Show gold";
}});
$("exportCsv").addEventListener("click",()=>exportRows("csv"));
$("exportJson").addEventListener("click",()=>exportRows("json"));
document.querySelectorAll("input[type=radio]").forEach(x=>x.addEventListener("change",()=>{{saveCurrent();render();}}));
$("notes").addEventListener("change",()=>{{saveCurrent();render();}});
$("reviewer").addEventListener("change",()=>{{saveCurrent();render();}});
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=REPO_ROOT / "frozen_outputs/minigrid/taskd-r/kimi-k2.5/score_per_row.jsonl",
    )
    parser.add_argument(
        "--exam",
        type=Path,
        default=REPO_ROOT / "datasets/minigrid/taskd-r/task_d_exam.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "human_review/kimi_taskdr_parser",
    )
    args = parser.parse_args()
    records = build_records(args.scores, args.exam)
    summary = {
        "items": len(records),
        "parser_changed": sum(row["parser_changed"] for row in records),
        "parse_modes": dict(Counter(row["corrected_mode"] for row in records)),
        "strict_rows_omitted": 11,
        "environments": dict(Counter(row["env_task"] for row in records)),
        "variants": dict(Counter(row["variant"] for row in records)),
        "source": str(args.scores.relative_to(REPO_ROOT)),
    }
    if summary["items"] != 133 or summary["parser_changed"] != 49:
        raise SystemExit(f"unexpected review population: {summary}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "review_sheet.csv", records)
    (args.output_dir / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "review_app.html").write_text(
        build_html(records, summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
