"""
XIMPAX Intelligence Engine — Stage 1
Runs Prompt 1 through Gemini 10 times, deduplicates, scores and outputs an HTML chart.
"""

import os
import re
import json
import time
import logging
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-1.5-pro"
NUM_RUNS = 10
RUNS_BETWEEN_SLEEP = 5          # Gemini free-tier rate limit guard
SLEEP_SECONDS = 15
TOP_N_FOR_STAGE2 = 10


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_prompt() -> str:
    company_profile = load_file(CONFIG_DIR / "company_profile.txt")
    icp_blueprint   = load_file(CONFIG_DIR / "icp_blueprint.txt")
    instructions    = load_file(CONFIG_DIR / "instructions.txt")
    prompt_template = load_file(PROMPTS_DIR / "prompt_1.txt")

    # Inject context documents into the prompt
    full_prompt = f"""
=== FIRM PROFILE DOCUMENT ===
{company_profile}

=== ICP BLUEPRINT ===
{icp_blueprint}

=== SITUATION DETECTION FRAMEWORK (INSTRUCTIONS) ===
{instructions}

=== TASK PROMPT ===
{prompt_template}
"""
    return full_prompt


# ── Gemini call ────────────────────────────────────────────────────────────────
def call_gemini(prompt: str, run_index: int) -> str:
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

    log.info(f"Run {run_index+1}/{NUM_RUNS} → calling Gemini …")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=8192,
        ),
    )
    return response.text


# ── Table parsing ──────────────────────────────────────────────────────────────
COLUMNS = [
    "Tier", "Company Name", "HQ Country", "Industry", "Revenue (USD)",
    "Situations Detected", "Classification", "Priority Score",
    "Key Evidence", "Sources"
]

def parse_markdown_table(raw: str) -> list[dict]:
    """Extract rows from a pipe-delimited markdown table."""
    rows = []
    in_table = False
    header_passed = False

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if in_table:
                break           # table ended
            continue

        in_table = True
        if re.match(r"^\|[-| :]+\|$", line):
            header_passed = True
            continue
        if not header_passed:
            continue            # still header row

        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 10:
            continue            # malformed row

        rows.append({
            "tier":             cells[0],
            "company":          cells[1],
            "hq_country":       cells[2],
            "industry":         cells[3],
            "revenue":          cells[4],
            "situations":       cells[5],
            "classification":   cells[6],
            "priority_score":   cells[7],
            "key_evidence":     cells[8],
            "sources":          cells[9],
        })
    return rows


def extract_total_score(score_str: str) -> int:
    """Parse 'RC: 4 | MP: 6 | SG: 2 | SCD: 3 = 15/40' → 15"""
    m = re.search(r"=\s*(\d+)\s*/\s*40", score_str)
    if m:
        return int(m.group(1))
    # Fallback: sum RC/MP/SG/SCD values
    parts = re.findall(r":\s*(\d+)", score_str)
    return sum(int(p) for p in parts) if parts else 0


# ── Aggregation ────────────────────────────────────────────────────────────────
def aggregate_runs(all_runs: list[list[dict]]) -> list[dict]:
    """
    Merge 10 runs into a single deduplicated ranked list.
    Companies appearing in more runs with higher scores rank higher.
    """
    seen: dict[str, dict] = {}        # canonical_name → best entry

    for run_rows in all_runs:
        for row in run_rows:
            name = row["company"].strip().upper()
            name = re.sub(r"\b(AG|SA|GMBH|LTD|PLC|INC|LLC|NV|BV|SE|SPA|SAS)\b", "", name).strip()
            score = extract_total_score(row["priority_score"])
            row["_score_int"] = score
            row["_frequency"] = 1

            if name not in seen:
                seen[name] = row
            else:
                existing = seen[name]
                existing["_frequency"] += 1
                if score > existing["_score_int"]:
                    row["_frequency"] = existing["_frequency"]
                    seen[name] = row

    # Sort: Tier 1 first, then by score desc, then by frequency desc
    merged = list(seen.values())
    merged.sort(key=lambda r: (
        0 if "1" in r["tier"] else 1,
        -r["_score_int"],
        -r["_frequency"],
    ))
    return merged


# ── HTML generation ────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XIMPAX Stage 1 Intelligence Report — {date}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; color: #1a1a2e; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; color: #1a1a2e; }}
  .subtitle {{ color: #555; font-size: 13px; margin-bottom: 20px; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  .stat-card {{ background: #fff; border-radius: 8px; padding: 12px 20px; box-shadow: 0 1px 4px rgba(0,0,0,.1); min-width: 130px; }}
  .stat-card .val {{ font-size: 28px; font-weight: 700; color: #2563eb; }}
  .stat-card .lbl {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: .5px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  thead {{ background: #1a1a2e; color: #fff; }}
  thead th {{ padding: 10px 8px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; white-space: nowrap; }}
  tbody tr {{ border-bottom: 1px solid #e8ecf0; }}
  tbody tr:hover {{ background: #f0f4ff; }}
  tbody td {{ padding: 9px 8px; vertical-align: top; }}
  .tier1 {{ border-left: 3px solid #2563eb; }}
  .tier2 {{ border-left: 3px solid #94a3b8; }}
  .badge {{ display: inline-block; padding: 2px 7px; border-radius: 12px; font-size: 10px; font-weight: 600; }}
  .confirmed {{ background: #dcfce7; color: #166534; }}
  .likely {{ background: #dbeafe; color: #1e40af; }}
  .unclear {{ background: #fef9c3; color: #854d0e; }}
  .not-present {{ background: #f1f5f9; color: #64748b; }}
  .score-bar-wrap {{ width: 100%; background: #e8ecf0; border-radius: 4px; height: 6px; margin-top: 4px; }}
  .score-bar {{ height: 6px; border-radius: 4px; background: linear-gradient(90deg,#2563eb,#7c3aed); }}
  .score-num {{ font-weight: 700; font-size: 13px; }}
  .freq {{ font-size: 10px; color: #64748b; }}
  a {{ color: #2563eb; text-decoration: none; font-size: 11px; }}
  a:hover {{ text-decoration: underline; }}
  .evidence {{ font-size: 11px; line-height: 1.5; }}
  .top10-marker {{ background: #fef3c7; }}
  .highlight-row td {{ background: #eff6ff !important; }}
  .run-info {{ margin-top: 16px; font-size: 11px; color: #888; }}
</style>
</head>
<body>
<h1>XIMPAX Market Intelligence — Stage 1 Report</h1>
<p class="subtitle">Generated: {date} | {num_runs} AI runs aggregated | Sorted by Priority Score</p>

<div class="stats">
  <div class="stat-card"><div class="val">{total}</div><div class="lbl">Companies Found</div></div>
  <div class="stat-card"><div class="val">{tier1}</div><div class="lbl">Tier 1 Leads</div></div>
  <div class="stat-card"><div class="val">{tier2}</div><div class="lbl">Tier 2 Prospects</div></div>
  <div class="stat-card"><div class="val">{top10}</div><div class="lbl">Going to Stage 2</div></div>
</div>

<table>
<thead>
  <tr>
    <th>#</th>
    <th>Tier</th>
    <th>Company</th>
    <th>HQ</th>
    <th>Industry</th>
    <th>Revenue</th>
    <th>Situations</th>
    <th>Classification</th>
    <th>Priority Score</th>
    <th>Frequency</th>
    <th>Key Evidence</th>
    <th>Sources</th>
  </tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
<p class="run-info">Top {top10} companies (highlighted) forwarded to Stage 2 deep scan.</p>
</body>
</html>"""


def classification_badge(cls: str) -> str:
    cls_l = cls.lower()
    if "confirmed" in cls_l:
        return f'<span class="badge confirmed">✓ Confirmed</span>'
    elif "likely" in cls_l:
        return f'<span class="badge likely">~ Likely</span>'
    elif "unclear" in cls_l:
        return f'<span class="badge unclear">? Unclear</span>'
    return f'<span class="badge not-present">— Not Present</span>'


def make_sources_html(sources: str) -> str:
    urls = re.findall(r"https?://[^\s,<>\"']+", sources)
    if urls:
        return "<br>".join(f'<a href="{u}" target="_blank">🔗 Source</a>' for u in urls)
    return sources[:120]


def build_html(rows: list[dict], num_runs: int) -> str:
    date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tier1_count = sum(1 for r in rows if "1" in r["tier"])
    tier2_count = len(rows) - tier1_count
    top_n = min(TOP_N_FOR_STAGE2, tier1_count)

    html_rows = []
    for i, row in enumerate(rows):
        is_top10 = i < top_n
        score = row["_score_int"]
        freq = row["_frequency"]
        tier_class = "tier1" if "1" in row["tier"] else "tier2"
        highlight = "highlight-row" if is_top10 else ""
        badge = f'<span style="font-weight:600;color:{"#2563eb" if "1" in row["tier"] else "#94a3b8"}">{"★ Tier 1" if "1" in row["tier"] else "Tier 2"}</span>'

        bar_pct = min(100, int(score / 40 * 100))
        score_html = f"""
          <span class="score-num">{score}/40</span>
          <div class="score-bar-wrap"><div class="score-bar" style="width:{bar_pct}%"></div></div>
          <small style="font-size:10px;color:#888">{row['priority_score']}</small>
        """

        situations = row["situations"].replace(",", "<br>")
        classification = classification_badge(row["classification"])

        evidence_html = row["key_evidence"].replace("•", "<br>•").replace("* ", "<br>• ")

        marker = "🥇 " if is_top10 else ""

        html_rows.append(f"""
        <tr class="{tier_class} {highlight}">
          <td><b>{marker}{i+1}</b></td>
          <td>{badge}</td>
          <td><b>{row['company']}</b></td>
          <td>{row['hq_country']}</td>
          <td>{row['industry']}</td>
          <td>{row['revenue']}</td>
          <td class="evidence">{situations}</td>
          <td>{classification}</td>
          <td>{score_html}</td>
          <td><span class="freq">Seen in {freq}/{num_runs} runs</span></td>
          <td class="evidence">{evidence_html[:300]}</td>
          <td>{make_sources_html(row['sources'])}</td>
        </tr>""")

    return HTML_TEMPLATE.format(
        date=date_str,
        num_runs=num_runs,
        total=len(rows),
        tier1=tier1_count,
        tier2=tier2_count,
        top10=top_n,
        rows="\n".join(html_rows),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    prompt = build_prompt()
    all_runs: list[list[dict]] = []
    raw_responses = []

    for i in range(NUM_RUNS):
        try:
            raw = call_gemini(prompt, i)
            raw_responses.append(raw)
            parsed = parse_markdown_table(raw)
            log.info(f"  → Parsed {len(parsed)} rows")
            if parsed:
                all_runs.append(parsed)
        except Exception as e:
            log.warning(f"Run {i+1} failed: {e}")

        if (i + 1) % RUNS_BETWEEN_SLEEP == 0 and i < NUM_RUNS - 1:
            log.info(f"Rate-limit sleep {SLEEP_SECONDS}s …")
            time.sleep(SLEEP_SECONDS)

    if not all_runs:
        raise RuntimeError("No runs produced parseable output — check API key and model access.")

    # Save raw responses for audit
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"stage1_raw_{ts}.json"
    raw_path.write_text(json.dumps(raw_responses, ensure_ascii=False, indent=2))
    log.info(f"Raw responses saved → {raw_path}")

    merged = aggregate_runs(all_runs)
    log.info(f"Merged into {len(merged)} unique companies")

    html = build_html(merged, NUM_RUNS)
    html_path = OUTPUT_DIR / f"stage1_chart_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Stage 1 HTML chart → {html_path}")

    # Save top-N as JSON for stage 2
    top_n = min(TOP_N_FOR_STAGE2, sum(1 for r in merged if "1" in r["tier"]))
    top_companies = merged[:top_n]
    stage2_input_path = OUTPUT_DIR / "stage2_input.json"
    stage2_input_path.write_text(
        json.dumps(top_companies, ensure_ascii=False, indent=2)
    )
    log.info(f"Top {top_n} companies saved for Stage 2 → {stage2_input_path}")
    return html_path, stage2_input_path


if __name__ == "__main__":
    main()
