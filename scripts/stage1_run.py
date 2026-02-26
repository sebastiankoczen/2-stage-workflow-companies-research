"""
XIMPAX Intelligence Engine — Stage 1
Two-call architecture:
  Call A → Gemini + Google Search grounding → prose research findings (real URLs)
  Call B → Gemini (no tools) → converts findings into the required markdown table

This separation is critical: Google Search grounding conflicts with table-only
output mode, so we must split research and formatting into separate calls.
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
CONFIG_DIR  = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
OUTPUT_DIR  = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL       = "gemini-2.0-flash"
NUM_RUNS           = 1      # ← set to 10 for production
RUNS_BETWEEN_SLEEP = 3
SLEEP_SECONDS      = 20
TOP_N_FOR_STAGE2   = 2      # ← set to 10 for production

REVENUE_MIN_B = 0.5
REVENUE_MAX_B = 15.0

# Date 12 months ago — used in prompts to enforce recency
CUTOFF_DATE = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
TODAY       = datetime.utcnow().strftime("%Y-%m-%d")


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


# ── CALL A: Research prompt ────────────────────────────────────────────────────
def build_research_prompt() -> str:
    """
    Call A prompt: ask Gemini (with Google Search) to find 20 companies
    and gather raw evidence for all 4 situations. Returns prose with real URLs.
    """
    company_profile = load_file(CONFIG_DIR / "company_profile.txt")
    icp_blueprint   = load_file(CONFIG_DIR / "icp_blueprint.txt")
    instructions    = load_file(CONFIG_DIR / "instructions.txt")

    return f"""
You are a supply chain market intelligence researcher. Today is {TODAY}.

Your task is to find 20 Swiss companies (or companies with strong Swiss presence) in
these industries: Pharmaceuticals & Life Sciences, MedTech & Medical Devices,
FMCG & Consumer Goods, Luxury Goods & Cosmetics, Food & Beverage,
Chemicals & Specialty Chemicals, Agribusiness, Packaging, Industrial Manufacturing,
Automotive Components, Energy & Utilities.

Revenue range: USD 0.5B to 15B MAXIMUM. Exclude: Roche, Novartis, Nestlé, ABB,
Zurich Insurance, Swiss Re, UBS (all >$15B revenue).

For EACH of the 20 companies, search for evidence of ALL FOUR situations below.
You must search for each situation independently — do not skip any.

SITUATION 1 — RESOURCE CONSTRAINTS
Search queries to run: "[company] supply chain procurement staffing shortage capacity 2024 2025"
and "[company] CPO supply chain director departure vacancy urgent hire 2024 2025"
Look for: leadership churn, hiring freezes, delayed ERP/IBP/S&OP projects, contractor reliance

SITUATION 2 — MARGIN PRESSURE
Search queries to run: "[company] EBITDA margin decline cost reduction restructuring 2024 2025"
and "[company] profit warning savings program procurement efficiency 2024 2025"
Look for: margin decline, layoffs, plant closures, quantified savings targets, SKU reduction

SITUATION 3 — SIGNIFICANT GROWTH
Search queries to run: "[company] acquisition M&A new plant capacity expansion 2024 2025"
and "[company] revenue growth supply chain scaling new market 2024 2025"
Look for: M&A activity, new factories, DC expansion, rapid hiring ramps, backlog pressure

SITUATION 4 — SUPPLY CHAIN DISRUPTION
Search queries to run: "[company] supply disruption shortage production halt logistics 2024 2025"
and "[company] recall quality crisis supplier failure inventory shock 2024 2025"
Look for: production stoppages, recalls, missed guidance due to supply, force majeure

RULES:
- Only use evidence dated after {CUTOFF_DATE} (last 12 months)
- Preferred sources: Reuters, Bloomberg, FT, earnings call transcripts, regulatory filings
- Forbidden sources: company websites, investor relations pages, vendor blogs
- For every piece of evidence, you MUST provide the actual URL and publication date
- If you cannot find a real URL, say "no evidence found" — do not fabricate links

=== FIRM PROFILE (for ICP matching) ===
{company_profile}

=== ICP BLUEPRINT ===
{icp_blueprint}

=== SITUATION DETECTION FRAMEWORK ===
{instructions}

OUTPUT FORMAT:
For each company, write a structured research block like this:

COMPANY: [Name]
HQ: [Country]
INDUSTRY: [Specific sub-industry]
REVENUE: [USD figure or Approx. band]

SITUATION 1 — RESOURCE CONSTRAINTS:
- Signal: [signal name] | Strength: STRONG/MEDIUM | Score: +2/+1
  Evidence: "[verbatim quote under 25 words]" 
  Date: YYYY-MM-DD | Source: [Publication name] | URL: [full https:// URL]
[repeat for each signal found, or write "No evidence found"]

SITUATION 2 — MARGIN PRESSURE:
[same format]

SITUATION 3 — SIGNIFICANT GROWTH:
[same format]

SITUATION 4 — SUPPLY CHAIN DISRUPTION:
[same format]

SCORES: RC:[n] | MP:[n] | SG:[n] | SCD:[n] | TOTAL:[n]/40
TIER: [Tier 1 or Tier 2]

---
[repeat for all 20 companies]
"""


# ── CALL B: Table formatting prompt ───────────────────────────────────────────
def build_format_prompt(research_findings: str) -> str:
    """
    Call B prompt: take the prose research findings from Call A and
    format them into the exact 10-column markdown table required.
    No tools, no internet — pure formatting.
    """
    prompt_template = load_file(PROMPTS_DIR / "prompt_1.txt")

    return f"""
You are a data formatting engine. Convert the research findings below into a 
markdown table. Do NOT do additional research. Do NOT change scores or evidence.
Just reformat exactly what is in the research findings into the table structure.

=== RESEARCH FINDINGS TO FORMAT ===
{research_findings}

=== TABLE FORMAT REQUIRED ===
{prompt_template}

CRITICAL FORMATTING RULES:
1. Output ONLY the markdown table — no text before or after
2. Every row must start with | and end with |
3. Use exactly these 10 columns: Tier | Company Name | HQ Country | Industry | Revenue (USD) | Situations Detected | Classification | Priority Score | Key Evidence | Sources
4. Priority Score format: RC: X | MP: X | SG: X | SCD: X = XX/40
5. For Sources column: only include URLs that appear in the research findings above — do not add, guess or modify any URL
6. Sort: Tier 1 companies first (by total score descending), then Tier 2
7. If research findings show "No evidence found" for a situation, score it 0
"""


# ── Gemini API calls ──────────────────────────────────────────────────────────
def gemini_research(prompt: str, run_index: int) -> str:
    """Call A: research with Google Search grounding enabled."""
    api_key = os.environ["GEMINI_API_KEY"]
    client  = genai.Client(api_key=api_key)
    log.info(f"Run {run_index+1}/{NUM_RUNS} → Call A: researching with Google Search …")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.3,
            max_output_tokens=16000,
        ),
    )
    text = response.text or ""
    log.info(f"  → Call A returned {len(text)} chars")
    return text


def gemini_format(research_findings: str, run_index: int) -> str:
    """Call B: format prose findings into table (no tools, no search)."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    log.info(f"Run {run_index+1}/{NUM_RUNS} → Call B: formatting into table …")

    format_prompt = build_format_prompt(research_findings)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=format_prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a data formatting engine. Output ONLY the markdown table. "
                "Start directly with the | character of the header row. "
                "Every row must start and end with |. "
                "Do not add any text before or after the table. "
                "Do not invent or modify any URLs — only use URLs from the input."
            ),
            temperature=0.1,
            max_output_tokens=16000,
        ),
    )
    text = response.text or ""
    log.info(f"  → Call B returned {len(text)} chars")
    return text


def run_once(run_index: int) -> tuple[str, str]:
    """Execute one full run: research → format. Returns (research_text, table_text)."""
    research_prompt  = build_research_prompt()
    research_text    = gemini_research(research_prompt, run_index)

    if not research_text.strip():
        raise RuntimeError("Call A returned empty response")

    # Small pause between the two calls
    time.sleep(5)

    table_text = gemini_format(research_text, run_index)
    return research_text, table_text


# ── Table parsing ──────────────────────────────────────────────────────────────
def parse_markdown_table(raw: str) -> list[dict]:
    raw        = re.sub(r"```[a-z]*", "", raw)
    lines      = [l.strip() for l in raw.splitlines()]
    pipe_lines = [l for l in lines if l.startswith("|")]

    if not pipe_lines:
        log.warning("No pipe-delimited lines found.")
        return []

    rows           = []
    header_skipped = False

    for line in pipe_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]

        if all(re.match(r"^[-:\s]+$", c) for c in cells if c):
            header_skipped = True
            continue
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
    m = re.search(r"=\s*(\d+)\s*/\s*40", score_str)
    if m:
        return int(m.group(1))
    parts = re.findall(r":\s*(\d+)", score_str)
    return sum(int(p) for p in parts) if parts else 0


def parse_revenue_billions(rev_str: str) -> float | None:
    s = rev_str.upper().replace(",", "")
    if any(x in s for x in ("UNKNOWN", "N/A")):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*B", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*M(?!A)", s)
    if m:
        return float(m.group(1)) / 1000
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        val = float(m.group(1))
        return val if val <= 500 else val / 1000
    return None


def revenue_in_range(rev_str: str) -> bool:
    val = parse_revenue_billions(rev_str)
    if val is None:
        return True
    return REVENUE_MIN_B <= val <= REVENUE_MAX_B


# ── Aggregation ────────────────────────────────────────────────────────────────
def aggregate_runs(all_runs: list[list[dict]]) -> list[dict]:
    seen: dict[str, dict] = {}

    for run_rows in all_runs:
        for row in run_rows:
            if not revenue_in_range(row["revenue"]):
                log.info(f"  ✗ Revenue filter: {row['company']} ({row['revenue']})")
                continue

            name  = row["company"].strip().upper()
            name  = re.sub(r"\b(AG|SA|GMBH|LTD|PLC|INC|LLC|NV|BV|SE|SPA|SAS)\b", "", name).strip()
            score = extract_total_score(row["priority_score"])
            row["_score_int"] = score
            row["_frequency"] = 1

            if name not in seen:
                seen[name] = row
            else:
                seen[name]["_frequency"] += 1
                if score > seen[name]["_score_int"]:
                    freq = seen[name]["_frequency"]
                    seen[name] = row
                    seen[name]["_frequency"] = freq

    merged = list(seen.values())
    merged.sort(key=lambda r: (
        0 if "1" in r["tier"] else 1,
        -r["_score_int"],
        -r["_frequency"],
    ))

    log.info(f"Aggregated: {len(merged)} unique companies after revenue filter")
    for r in merged[:5]:
        log.info(f"  → {r['company']} | {r['_score_int']}/40 | {r['tier']}")
    return merged


# ── HTML generation ────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XIMPAX Stage 1 — {date}</title>
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
  .confirmed {{ background: #14532d; color: #86efac; }}
  .likely    {{ background: #1e3a5f; color: #93c5fd; }}
  .unclear   {{ background: #422006; color: #fcd34d; }}
  .not-present {{ background: #1f2937; color: #6b7280; border: 1px solid #374151; }}
  .score-bar-wrap {{ width: 100%; background: #374151; border-radius: 4px; height: 6px; margin-top: 4px; }}
  .score-bar {{ height: 6px; border-radius: 4px; background: linear-gradient(90deg,#2563eb,#7c3aed); }}
  .score-num {{ font-weight: 700; font-size: 13px; color: #60a5fa; }}
  .freq {{ font-size: 10px; color: #6b7280; }}
  a {{ color: #60a5fa; text-decoration: none; font-size: 11px; }}
  a:hover {{ text-decoration: underline; }}
  .evidence {{ font-size: 11px; line-height: 1.5; color: #9ca3af; }}
  .highlight-row td {{ background: #1e3a5f !important; }}
  .run-info {{ margin-top: 16px; font-size: 11px; color: #6b7280; }}
  .company-name {{ color: #f9fafb; font-weight: 600; }}
</style>
</head>
<body>
<h1>XIMPAX Market Intelligence — Stage 1 Report</h1>
<p class="subtitle">Generated: {date} | {num_runs} AI runs | Sorted by Priority Score ↓</p>
<div class="stats">
  <div class="stat-card"><div class="val">{total}</div><div class="lbl">Companies</div></div>
  <div class="stat-card"><div class="val">{tier1}</div><div class="lbl">Tier 1 Leads</div></div>
  <div class="stat-card"><div class="val">{tier2}</div><div class="lbl">Tier 2</div></div>
  <div class="stat-card"><div class="val">{top10}</div><div class="lbl">→ Stage 2</div></div>
</div>
<table>
<thead>
  <tr>
    <th>#</th><th>Tier</th><th>Company</th><th>HQ</th><th>Industry</th>
    <th>Revenue</th><th>Situations</th><th>Classification</th>
    <th>Priority Score</th><th>Runs</th><th>Key Evidence</th><th>Sources</th>
  </tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
<p class="run-info">🥇 Highlighted rows = Top {top10} companies forwarded to Stage 2 (Tier 1 only, highest scores first).</p>
</body>
</html>"""


def classification_badge(cls: str) -> str:
    c = cls.lower()
    if "confirmed" in c: return '<span class="badge confirmed">✓ Confirmed</span>'
    if "likely"    in c: return '<span class="badge likely">~ Likely</span>'
    if "unclear"   in c: return '<span class="badge unclear">? Unclear</span>'
    return '<span class="badge not-present">— N/P</span>'


def make_sources_html(sources: str) -> str:
    urls = re.findall(r"https?://[^\s,<>\"'\]]+", sources)
    if urls:
        return "<br>".join(f'<a href="{u}" target="_blank">🔗</a>' for u in urls[:3])
    return f'<span style="color:#6b7280;font-size:10px">{sources[:80]}</span>'


def build_html(rows: list[dict], num_runs: int) -> str:
    date_str    = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tier1_count = sum(1 for r in rows if "1" in r["tier"])
    tier2_count = len(rows) - tier1_count
    tier1_rows  = [r for r in rows if "1" in r["tier"]]
    top_n       = min(TOP_N_FOR_STAGE2, len(tier1_rows))
    top_set     = set(id(r) for r in tier1_rows[:top_n])

    html_rows = []
    for i, row in enumerate(rows):
        is_top     = id(row) in top_set
        score      = row["_score_int"]
        freq       = row["_frequency"]
        tier_class = "tier1" if "1" in row["tier"] else "tier2"
        highlight  = "highlight-row" if is_top else ""
        tier_color = "#60a5fa" if "1" in row["tier"] else "#4b5563"
        tier_label = "★ Tier 1" if "1" in row["tier"] else "Tier 2"
        bar_pct    = min(100, int(score / 40 * 100))
        marker     = "🥇 " if is_top else ""

        score_html = (
            f'<span class="score-num">{score}/40</span>'
            f'<div class="score-bar-wrap"><div class="score-bar" style="width:{bar_pct}%"></div></div>'
            f'<small style="font-size:10px;color:#6b7280">{row["priority_score"]}</small>'
        )
        situations     = row["situations"].replace(",", "<br>").replace(";", "<br>")
        classification = classification_badge(row["classification"])
        evidence_html  = row["key_evidence"].replace("•", "<br>•").replace("* ", "<br>• ")

        html_rows.append(f"""
        <tr class="{tier_class} {highlight}">
          <td><b style="color:#f9fafb">{marker}{i+1}</b></td>
          <td><span style="font-weight:600;color:{tier_color}">{tier_label}</span></td>
          <td class="company-name">{row['company']}</td>
          <td>{row['hq_country']}</td>
          <td>{row['industry']}</td>
          <td>{row['revenue']}</td>
          <td class="evidence">{situations}</td>
          <td>{classification}</td>
          <td>{score_html}</td>
          <td><span class="freq">{freq}/{num_runs}</span></td>
          <td class="evidence">{evidence_html[:280]}</td>
          <td>{make_sources_html(row['sources'])}</td>
        </tr>""")

    return HTML_TEMPLATE.format(
        date=date_str, num_runs=num_runs,
        total=len(rows), tier1=tier1_count, tier2=tier2_count, top10=top_n,
        rows="\n".join(html_rows),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    all_runs:       list[list[dict]] = []
    all_research:   list[str]        = []
    all_raw_tables: list[str]        = []

    for i in range(NUM_RUNS):
        try:
            research_text, table_text = run_once(i)
            all_research.append(research_text)
            all_raw_tables.append(table_text)

            parsed = parse_markdown_table(table_text)
            log.info(f"  → Parsed {len(parsed)} companies from run {i+1}")
            if len(parsed) == 0:
                log.warning(f"  → 0 rows. Table preview:\n{table_text[:1000]}")
            if parsed:
                all_runs.append(parsed)

        except Exception as e:
            log.warning(f"Run {i+1} failed: {e}")

        if (i + 1) % RUNS_BETWEEN_SLEEP == 0 and i < NUM_RUNS - 1:
            log.info(f"Sleeping {SLEEP_SECONDS}s (rate limit) …")
            time.sleep(SLEEP_SECONDS)

    if not all_runs:
        raise RuntimeError("No runs produced parseable output.")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save raw research + tables for audit
    (OUTPUT_DIR / f"stage1_research_{ts}.json").write_text(
        json.dumps({"research": all_research, "tables": all_raw_tables},
                   ensure_ascii=False, indent=2)
    )

    merged    = aggregate_runs(all_runs)
    html      = build_html(merged, NUM_RUNS)
    html_path = OUTPUT_DIR / f"stage1_chart_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Stage 1 chart → {html_path}")

    # Top N for Stage 2: Tier 1 only, sorted by score DESC
    tier1_only    = [r for r in merged if "1" in r["tier"]]
    top_n         = min(TOP_N_FOR_STAGE2, len(tier1_only))
    top_companies = tier1_only[:top_n]

    log.info(f"Stage 2 input ({top_n} companies):")
    for r in top_companies:
        log.info(f"  → {r['company']} | {r['_score_int']}/40")

    stage2_path = OUTPUT_DIR / "stage2_input.json"
    stage2_path.write_text(json.dumps(top_companies, ensure_ascii=False, indent=2))
    log.info(f"Stage 2 input saved → {stage2_path}")

    return html_path, stage2_path


if __name__ == "__main__":
    main()
