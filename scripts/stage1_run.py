"""
XIMPAX Intelligence Engine — Stage 1
Runs Prompt 1 through Gemini N times, deduplicates, scores and outputs an HTML chart.
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
CONFIG_DIR  = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
OUTPUT_DIR  = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL       = "gemini-2.0-flash"
NUM_RUNS           = 1          # ← set to 10 for production
RUNS_BETWEEN_SLEEP = 5
SLEEP_SECONDS      = 15
TOP_N_FOR_STAGE2   = 2          # ← set to 10 for production

# ── Revenue filter ─────────────────────────────────────────────────────────────
# Companies above this threshold (USD billions) are excluded after parsing.
# Roche ~$62B, Novartis ~$45B, BASF ~$68B — all should be blocked by this.
REVENUE_MIN_B = 0.5
REVENUE_MAX_B = 15.0


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_prompt() -> str:
    company_profile = load_file(CONFIG_DIR / "company_profile.txt")
    icp_blueprint   = load_file(CONFIG_DIR / "icp_blueprint.txt")
    instructions    = load_file(CONFIG_DIR / "instructions.txt")
    prompt_template = load_file(PROMPTS_DIR / "prompt_1.txt")

    full_prompt = f"""
=== FIRM PROFILE DOCUMENT ===
{company_profile}

=== ICP BLUEPRINT ===
{icp_blueprint}

=== SITUATION DETECTION FRAMEWORK (INSTRUCTIONS) ===
{instructions}

=== CAMPAIGN FILTERS — STRICTLY ENFORCED ===
These filters are HARD REQUIREMENTS. Any company that does not meet ALL of them must be excluded entirely.

- Target Region: Switzerland (HQ or primary operating entity must be in Switzerland)
- Revenue range: USD 0.5 billion to USD 15 billion MAXIMUM.
  DO NOT include companies with revenue above USD 15 billion.
  This explicitly excludes: Roche, Novartis, Nestlé, ABB, Zurich Insurance, Swiss Re, UBS, Credit Suisse, BASF, or any other company with revenue above $15B.
- Industries: Pharmaceuticals and Life Sciences, MedTech and Medical Devices, FMCG and Consumer Goods,
  Luxury Goods and Cosmetics, Food and Beverage Manufacturing, Chemicals and Specialty Chemicals,
  Agribusiness, Packaging, Industrial Manufacturing, Automotive Components, Energy and Utilities
- Evidence recency: last 12 months only

Before including any company, verify its revenue is between $0.5B and $15B USD. If uncertain, write "Approx." + range. If you cannot confirm it is under $15B, exclude it.

=== TASK PROMPT ===
{prompt_template}
"""
    return full_prompt


# ── Gemini call ────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are a supply chain market intelligence engine. "
    "When asked to produce a table, output ONLY the markdown table — no preamble, "
    "no acknowledgement, no explanation before or after the table. "
    "Start your response with the | character of the header row. "
    "Every single row including the header must start AND end with a | character. "
    "Do not truncate — include every row. "
    "Revenue filter is a hard rule: exclude any company with revenue above USD 15 billion."
)

def call_gemini(prompt: str, run_index: int) -> str:
    api_key = os.environ["GEMINI_API_KEY"]
    client  = genai.Client(api_key=api_key)

    log.info(f"Run {run_index+1}/{NUM_RUNS} → calling Gemini …")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.4,
            max_output_tokens=16000,
        ),
    )
    return response.text


# ── Revenue guard ──────────────────────────────────────────────────────────────
def parse_revenue_billions(rev_str: str) -> float | None:
    """
    Try to extract a numeric revenue figure in USD billions from the string.
    Returns None if it cannot be parsed confidently.
    Examples handled:
      "$8.2B"  "~$12B"  "Approx. $5–10B"  "CHF 6.4B"  "EUR 3.2B"  "12,400M"
    """
    s = rev_str.upper().replace(",", "")

    # If explicitly marked unknown, skip filtering
    if "UNKNOWN" in s or "N/A" in s:
        return None

    # Try to find a number followed by B (billions)
    m = re.search(r"(\d+(?:\.\d+)?)\s*B", s)
    if m:
        return float(m.group(1))

    # Try millions: 12400M → 12.4B
    m = re.search(r"(\d+(?:\.\d+)?)\s*M", s)
    if m:
        return float(m.group(1)) / 1000

    # Plain number — interpret as billions if ≤ 200, else millions
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        val = float(m.group(1))
        return val if val <= 200 else val / 1000

    return None


def revenue_in_range(rev_str: str) -> bool:
    """Returns True if revenue is within [REVENUE_MIN_B, REVENUE_MAX_B] or unknown."""
    val = parse_revenue_billions(rev_str)
    if val is None:
        return True   # Can't determine — keep it, let analyst decide
    return REVENUE_MIN_B <= val <= REVENUE_MAX_B


# ── Table parsing ──────────────────────────────────────────────────────────────
def parse_markdown_table(raw: str) -> list[dict]:
    """Robust parser — handles prose before/after table, code fences, missing trailing pipes."""
    raw = re.sub(r"```[a-z]*", "", raw)

    lines      = [l.strip() for l in raw.splitlines()]
    pipe_lines = [l for l in lines if l.startswith("|")]

    if not pipe_lines:
        log.warning("No pipe-delimited lines found in response at all.")
        return []

    rows           = []
    header_skipped = False

    for line in pipe_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]

        # Skip separator rows  |---|---|
        if all(re.match(r"^[-:\s]+$", c) for c in cells if c):
            header_skipped = True
            continue

        # First non-separator row = header — skip it
        if not header_skipped:
            header_skipped = True
            continue

        if len(cells) < 5:
            continue

        while len(cells) < 10:
            cells.append("")

        if cells[0].strip().lower() in ("tier", "col a", "column a", "#", ""):
            continue

        rows.append({
            "tier":           cells[0],
            "company":        cells[1],
            "hq_country":     cells[2],
            "industry":       cells[3],
            "revenue":        cells[4],
            "situations":     cells[5],
            "classification": cells[6],
            "priority_score": cells[7],
            "key_evidence":   cells[8],
            "sources":        cells[9],
        })

    return rows


def extract_total_score(score_str: str) -> int:
    """Parse 'RC: 4 | MP: 6 | SG: 2 | SCD: 3 = 15/40' → 15"""
    m = re.search(r"=\s*(\d+)\s*/\s*40", score_str)
    if m:
        return int(m.group(1))
    parts = re.findall(r":\s*(\d+)", score_str)
    return sum(int(p) for p in parts) if parts else 0


# ── Aggregation + filtering ────────────────────────────────────────────────────
def aggregate_runs(all_runs: list[list[dict]]) -> list[dict]:
    """
    1. Merge all runs, deduplicate by company name.
    2. Filter out companies outside the revenue range.
    3. Sort: Tier 1 first, then by score DESC, then by frequency DESC.
       This guarantees the highest-signal companies are at the top for Stage 2.
    """
    seen: dict[str, dict] = {}

    for run_rows in all_runs:
        for row in run_rows:
            # Revenue guard — skip oversized companies immediately
            if not revenue_in_range(row["revenue"]):
                log.info(f"  ✗ Excluded by revenue filter: {row['company']} ({row['revenue']})")
                continue

            name  = row["company"].strip().upper()
            name  = re.sub(r"\b(AG|SA|GMBH|LTD|PLC|INC|LLC|NV|BV|SE|SPA|SAS)\b", "", name).strip()
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

    # ── Sort: Tier 1 by score DESC, then Tier 2 by score DESC ─────────────────
    # This is the key sort that determines what goes to Stage 2.
    merged = list(seen.values())
    merged.sort(key=lambda r: (
        0 if "1" in r["tier"] else 1,   # Tier 1 before Tier 2
        -r["_score_int"],               # Highest score first
        -r["_frequency"],               # Most frequent across runs first
    ))

    log.info(f"After revenue filter + dedup: {len(merged)} companies")
    if merged:
        log.info("Top 5 by score: " + ", ".join(
            f"{r['company']} ({r['_score_int']}pts)" for r in merged[:5]
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
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #111827; color: #e5e7eb; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; color: #f9fafb; }}
  .subtitle {{ color: #9ca3af; font-size: 13px; margin-bottom: 20px; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  .stat-card {{ background: #1f2937; border-radius: 8px; padding: 12px 20px; border: 1px solid #374151; min-width: 130px; }}
  .stat-card .val {{ font-size: 28px; font-weight: 700; color: #60a5fa; }}
  .stat-card .lbl {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .5px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; background: #1f2937; border-radius: 8px; overflow: hidden; border: 1px solid #374151; }}
  thead {{ background: #0f172a; color: #f1f5f9; }}
  thead th {{ padding: 10px 8px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; white-space: nowrap; border-bottom: 2px solid #2563eb; }}
  tbody tr {{ border-bottom: 1px solid #374151; }}
  tbody tr:hover {{ background: #273548; }}
  tbody td {{ padding: 9px 8px; vertical-align: top; color: #d1d5db; }}
  .tier1 {{ border-left: 3px solid #2563eb; }}
  .tier2 {{ border-left: 3px solid #4b5563; }}
  .badge {{ display: inline-block; padding: 2px 7px; border-radius: 12px; font-size: 10px; font-weight: 600; }}
  .confirmed {{ background: #14532d; color: #86efac; border: 1px solid #166534; }}
  .likely {{ background: #1e3a5f; color: #93c5fd; border: 1px solid #1d4ed8; }}
  .unclear {{ background: #422006; color: #fcd34d; border: 1px solid #92400e; }}
  .not-present {{ background: #1f2937; color: #6b7280; border: 1px solid #374151; }}
  .score-bar-wrap {{ width: 100%; background: #374151; border-radius: 4px; height: 6px; margin-top: 4px; }}
  .score-bar {{ height: 6px; border-radius: 4px; background: linear-gradient(90deg,#2563eb,#7c3aed); }}
  .score-num {{ font-weight: 700; font-size: 13px; color: #60a5fa; }}
  .freq {{ font-size: 10px; color: #6b7280; }}
  a {{ color: #60a5fa; text-decoration: none; font-size: 11px; }}
  a:hover {{ text-decoration: underline; color: #93c5fd; }}
  .evidence {{ font-size: 11px; line-height: 1.5; color: #9ca3af; }}
  .highlight-row td {{ background: #1e3a5f !important; }}
  .run-info {{ margin-top: 16px; font-size: 11px; color: #6b7280; }}
  .company-name {{ color: #f9fafb; font-weight: 600; }}
</style>
</head>
<body>
<h1>XIMPAX Market Intelligence — Stage 1 Report</h1>
<p class="subtitle">Generated: {date} | {num_runs} AI runs aggregated | Sorted by Priority Score descending</p>

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
<p class="run-info">★ Top {top10} companies (highlighted in blue) forwarded to Stage 2 deep scan — selected by highest Priority Score among Tier 1 only.</p>
</body>
</html>"""


def classification_badge(cls: str) -> str:
    cls_l = cls.lower()
    if "confirmed" in cls_l:
        return '<span class="badge confirmed">✓ Confirmed</span>'
    elif "likely" in cls_l:
        return '<span class="badge likely">~ Likely</span>'
    elif "unclear" in cls_l:
        return '<span class="badge unclear">? Unclear</span>'
    return '<span class="badge not-present">— Not Present</span>'


def make_sources_html(sources: str) -> str:
    urls = re.findall(r"https?://[^\s,<>\"']+", sources)
    if urls:
        return "<br>".join(f'<a href="{u}" target="_blank">🔗 Source</a>' for u in urls)
    return f'<span style="color:#6b7280">{sources[:120]}</span>'


def build_html(rows: list[dict], num_runs: int) -> str:
    date_str    = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tier1_count = sum(1 for r in rows if "1" in r["tier"])
    tier2_count = len(rows) - tier1_count

    # Top N = highest-scored Tier 1 companies only
    tier1_rows = [r for r in rows if "1" in r["tier"]]
    top_n      = min(TOP_N_FOR_STAGE2, len(tier1_rows))
    top_set    = set(id(r) for r in tier1_rows[:top_n])

    html_rows = []
    for i, row in enumerate(rows):
        is_top = id(row) in top_set
        score  = row["_score_int"]
        freq   = row["_frequency"]

        tier_class = "tier1" if "1" in row["tier"] else "tier2"
        highlight  = "highlight-row" if is_top else ""
        tier_label = '★ Tier 1' if "1" in row["tier"] else "Tier 2"
        tier_color = "#60a5fa" if "1" in row["tier"] else "#4b5563"
        badge      = f'<span style="font-weight:600;color:{tier_color}">{tier_label}</span>'

        bar_pct    = min(100, int(score / 40 * 100))
        score_html = f"""
          <span class="score-num">{score}/40</span>
          <div class="score-bar-wrap"><div class="score-bar" style="width:{bar_pct}%"></div></div>
          <small style="font-size:10px;color:#6b7280">{row['priority_score']}</small>
        """

        situations     = row["situations"].replace(",", "<br>")
        classification = classification_badge(row["classification"])
        evidence_html  = row["key_evidence"].replace("•", "<br>•").replace("* ", "<br>• ")
        marker         = "🥇 " if is_top else ""

        html_rows.append(f"""
        <tr class="{tier_class} {highlight}">
          <td><b style="color:#f9fafb">{marker}{i+1}</b></td>
          <td>{badge}</td>
          <td class="company-name">{row['company']}</td>
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
    prompt    = build_prompt()
    all_runs: list[list[dict]] = []
    raw_responses = []

    for i in range(NUM_RUNS):
        try:
            raw = call_gemini(prompt, i)
            raw_responses.append(raw)
            parsed = parse_markdown_table(raw)
            log.info(f"  → Parsed {len(parsed)} rows (response: {len(raw)} chars)")
            if len(parsed) == 0:
                log.warning(f"  → 0 rows parsed. Raw output preview:\n{raw[:3000]}")
            if parsed:
                all_runs.append(parsed)
        except Exception as e:
            log.warning(f"Run {i+1} failed: {e}")

        if (i + 1) % RUNS_BETWEEN_SLEEP == 0 and i < NUM_RUNS - 1:
            log.info(f"Rate-limit sleep {SLEEP_SECONDS}s …")
            time.sleep(SLEEP_SECONDS)

    if not all_runs:
        raise RuntimeError("No runs produced parseable output — check API key and model access.")

    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"stage1_raw_{ts}.json"
    raw_path.write_text(json.dumps(raw_responses, ensure_ascii=False, indent=2))
    log.info(f"Raw responses saved → {raw_path}")

    merged = aggregate_runs(all_runs)
    log.info(f"Merged into {len(merged)} unique companies after revenue filter")

    html      = build_html(merged, NUM_RUNS)
    html_path = OUTPUT_DIR / f"stage1_chart_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Stage 1 HTML chart → {html_path}")

    # ── Top N for Stage 2: Tier 1 only, already sorted by score DESC ──────────
    tier1_only  = [r for r in merged if "1" in r["tier"]]
    top_n       = min(TOP_N_FOR_STAGE2, len(tier1_only))
    top_companies = tier1_only[:top_n]   # these are already highest-scored

    log.info(f"Top {top_n} Tier 1 companies for Stage 2 (by score):")
    for r in top_companies:
        log.info(f"  → {r['company']} | Score: {r['_score_int']}/40")

    stage2_input_path = OUTPUT_DIR / "stage2_input.json"
    stage2_input_path.write_text(
        json.dumps(top_companies, ensure_ascii=False, indent=2)
    )
    log.info(f"Stage 2 input saved → {stage2_input_path}")
    return html_path, stage2_input_path


if __name__ == "__main__":
    main()
