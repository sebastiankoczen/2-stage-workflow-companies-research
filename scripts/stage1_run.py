"""
XIMPAX Intelligence Engine — Stage 1

Architecture: single Gemini call per run, no Google Search tool.
Gemini 2.0 Flash has strong knowledge of Swiss industrial/pharma/FMCG companies
and their 2025 situations — sufficient for Stage 1 discovery.
Real URL verification happens in Stage 2 via live search.

10 runs × 20 companies = up to 200 raw entries → aggregated + deduped → top N to Stage 2.
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

ROOT        = Path(__file__).resolve().parent.parent
CONFIG_DIR  = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
OUTPUT_DIR  = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL       = "gemini-2.0-flash"
NUM_RUNS           = 1       # ← set to 10 for production
RUNS_BETWEEN_SLEEP = 3
SLEEP_SECONDS      = 20
TOP_N_FOR_STAGE2   = 2       # ← set to 10 for production
REVENUE_MIN_B      = 0.5
REVENUE_MAX_B      = 15.0

CUTOFF_DATE = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
TODAY       = datetime.utcnow().strftime("%Y-%m-%d")


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt() -> str:
    company_profile = load_file(CONFIG_DIR / "company_profile.txt")
    icp_blueprint   = load_file(CONFIG_DIR / "icp_blueprint.txt")
    instructions    = load_file(CONFIG_DIR / "instructions.txt")
    prompt_template = load_file(PROMPTS_DIR / "prompt_1.txt")

    return f"""
You are a senior supply chain market intelligence analyst. Today is {TODAY}.

=== FIRM PROFILE ===
{company_profile}

=== ICP BLUEPRINT ===
{icp_blueprint}

=== SITUATION DETECTION FRAMEWORK ===
{instructions}

=== HARD CAMPAIGN FILTERS ===
Region: Switzerland or companies with strong Swiss operational presence (within ~200km)
Revenue: USD {REVENUE_MIN_B}B minimum — USD {REVENUE_MAX_B}B MAXIMUM (hard cap)
EXCLUDED (revenue >$15B): Roche, Novartis, Nestlé, ABB, Zurich Insurance, Swiss Re, UBS, Credit Suisse
INCLUDED examples (revenue OK): Givaudan (~$7B ✓), Lonza (~$6B ✓), Sika (~$11B ✓),
  Sonova (~$4B ✓), Straumann (~$2B ✓), Tecan (~$1B ✓), Georg Fischer (~$4B ✓),
  Huber+Suhner (~$1B ✓), Emmi (~$4B ✓), Lindt & Sprüngli (~$5B ✓),
  Orior (~$0.7B ✓), Dätwyler (~$1.5B ✓), Bossard (~$0.6B ✓)
Industries: Pharma & Life Sciences, MedTech, FMCG, Luxury & Cosmetics,
  Food & Beverage, Chemicals, Agribusiness, Packaging, Industrial Manufacturing,
  Automotive Components, Energy & Utilities

=== MANDATORY EVALUATION RULES ===
You MUST evaluate ALL FOUR situations for every company. Do not skip any situation.
Score each situation independently — do not let one situation dominate.

For EVERY company you evaluate, explicitly work through these signal checks:

SITUATION 1 — RESOURCE CONSTRAINTS: Does this company show...
  ✓ Key SC/Procurement initiatives delayed due to lack of internal capacity?
  ✓ Leadership churn: CPO, VP Supply Chain, Head of Planning departed/replaced?
  ✓ Management explicitly mentioning "resource constraints" or "bandwidth" issues?
  ✓ High volume of SC/Procurement job vacancies (planners, buyers, logistics)?
  ✓ Contractor/interim reliance explicitly described for SC work?

SITUATION 2 — MARGIN PRESSURE: Does this company show...
  ✓ EBITDA or gross margin decline reported as structural (not one-off)?
  ✓ Quantified cost-out/savings program announced with targets?
  ✓ Guidance downgrade due to cost inflation or pricing pressure?
  ✓ Restructuring, plant closure, or SKU rationalization tied to margins?
  ✓ Procurement-driven savings program with specific targets?

SITUATION 3 — SIGNIFICANT GROWTH: Does this company show...
  ✓ M&A activity requiring supply chain integration?
  ✓ New production facility, DC, or capacity expansion announced?
  ✓ Revenue growth outpacing operational infrastructure?
  ✓ Explicit hiring ramp for SC execution roles due to growth?
  ✓ Entry into new geographies/channels requiring operational redesign?

SITUATION 4 — SUPPLY CHAIN DISRUPTION: Does this company show...
  ✓ Production shutdown, recall, or quality crisis affecting supply?
  ✓ Missed sales/guidance explicitly due to supply disruption?
  ✓ Supplier failure, force majeure, or logistics disruption with material impact?
  ✓ Crisis stabilization actions (re-sourcing, task force, allocation controls)?
  ✓ Inventory shock linked to disruption?

Use this scoring:
  STRONG signal confirmed = +2 pts | MEDIUM signal confirmed = +1 pt
  Max 10 pts per situation | Total max 40 pts
  CONFIRMED = 7-10 pts + ≥2 STRONG | LIKELY = 4-6 pts + ≥1 STRONG
  UNCLEAR = 2-3 pts | NOT PRESENT = 0-1 pts

TIER 1 = ICP match + at least one situation CONFIRMED or LIKELY
TIER 2 = ICP match but all situations UNCLEAR or NOT PRESENT

=== TASK ===
{prompt_template}

=== OUTPUT RULES (CRITICAL) ===
- Output ONLY the markdown table — no text before or after
- Start directly with | Tier | Company Name | ...
- Every row must start AND end with |
- Include EXACTLY 20 companies (mix of Tier 1 and Tier 2)
- You have strong knowledge of Swiss companies from 2024 news — use it fully
- For each company, use your best knowledge of their 2024 annual results,
  restructuring announcements, M&A activity, supply chain news
- Priority Score format: RC: X | MP: X | SG: X | SCD: X = XX/40
- For Sources: cite real publication names (Reuters, Bloomberg, FT, Seeking Alpha,
  earnings call Q3/Q4 2024, annual report 2024) — do not fabricate specific URLs
  (Stage 2 will verify URLs; here just cite the source type and date)
"""


# ── Gemini call ────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are a supply chain market intelligence analyst with deep knowledge of "
    "Swiss and European industrial, pharma, FMCG and chemical companies. "
    "You have comprehensive knowledge of 2024 corporate events: earnings results, "
    "restructurings, M&A, supply chain news, management changes. "
    "Output ONLY the markdown table. Start with the | character of the header row. "
    "Every row must start AND end with |. Output all 20 rows. Do not truncate."
)

def call_gemini(prompt: str, run_index: int) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    log.info(f"Run {run_index+1}/{NUM_RUNS} → calling Gemini (knowledge-based, no search) …")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.5,        # slight variation across runs for diversity
            max_output_tokens=16000,
        ),
    )
    text = response.text or ""
    log.info(f"  → {len(text)} chars returned")
    if len(text) < 500:
        log.warning(f"  → Very short response! Preview: {text[:300]}")
    return text


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
    for r in merged[:8]:
        log.info(f"  {r['company']:30s} | {r['_score_int']:2d}/40 | {r['tier']} | seen {r['_frequency']}x")
    return merged


# ── HTML output ────────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XIMPAX Stage 1 — {date}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #111827; color: #e5e7eb; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; color: #f9fafb; }}
  .sub {{ color: #9ca3af; font-size: 13px; margin-bottom: 20px; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  .sc {{ background: #1f2937; border-radius: 8px; padding: 12px 20px; border: 1px solid #374151; min-width: 130px; }}
  .sc .v {{ font-size: 28px; font-weight: 700; color: #60a5fa; }}
  .sc .l {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .5px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; background: #1f2937; border-radius: 8px; overflow: hidden; border: 1px solid #374151; }}
  thead {{ background: #0f172a; }}
  thead th {{ padding: 10px 8px; text-align: left; font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: .5px; white-space: nowrap; border-bottom: 2px solid #2563eb; }}
  tbody tr {{ border-bottom: 1px solid #374151; }}
  tbody tr:hover {{ background: #273548; }}
  tbody td {{ padding: 9px 8px; vertical-align: top; color: #d1d5db; }}
  .t1 {{ border-left: 3px solid #2563eb; }}
  .t2 {{ border-left: 3px solid #4b5563; }}
  .b {{ display: inline-block; padding: 2px 7px; border-radius: 12px; font-size: 10px; font-weight: 600; }}
  .bc {{ background:#14532d; color:#86efac; }}
  .bl {{ background:#1e3a5f; color:#93c5fd; }}
  .bu {{ background:#422006; color:#fcd34d; }}
  .bn {{ background:#1f2937; color:#6b7280; border:1px solid #374151; }}
  .sbw {{ width:100%; background:#374151; border-radius:4px; height:6px; margin-top:4px; }}
  .sb  {{ height:6px; border-radius:4px; background:linear-gradient(90deg,#2563eb,#7c3aed); }}
  .sn  {{ font-weight:700; font-size:13px; color:#60a5fa; }}
  a    {{ color:#60a5fa; text-decoration:none; font-size:11px; }}
  a:hover {{ text-decoration:underline; }}
  .ev  {{ font-size:11px; line-height:1.5; color:#9ca3af; }}
  .hl td {{ background:#1e3a5f !important; }}
  .cn  {{ color:#f9fafb; font-weight:600; }}
  .note {{ margin-top:14px; font-size:11px; color:#6b7280; }}
</style>
</head>
<body>
<h1>XIMPAX Market Intelligence — Stage 1 Report</h1>
<p class="sub">Generated: {date} | {num_runs} AI runs | {total} companies found | Sorted by Priority Score ↓</p>
<div class="stats">
  <div class="sc"><div class="v">{total}</div><div class="l">Companies</div></div>
  <div class="sc"><div class="v">{tier1}</div><div class="l">Tier 1 Leads</div></div>
  <div class="sc"><div class="v">{tier2}</div><div class="l">Tier 2</div></div>
  <div class="sc"><div class="v">{top10}</div><div class="l">→ Stage 2</div></div>
</div>
<table>
<thead>
  <tr>
    <th>#</th><th>Tier</th><th>Company</th><th>HQ</th><th>Industry</th>
    <th>Revenue</th><th>Situations</th><th>Classification</th>
    <th>Score</th><th>Runs</th><th>Key Evidence</th><th>Sources</th>
  </tr>
</thead>
<tbody>{rows}</tbody>
</table>
<p class="note">🥇 Highlighted = Top {top10} Tier 1 companies forwarded to Stage 2 for deep research with live web search.</p>
</body>
</html>"""


def badge(cls: str) -> str:
    c = cls.lower()
    if "confirmed" in c: return '<span class="b bc">✓ Confirmed</span>'
    if "likely"    in c: return '<span class="b bl">~ Likely</span>'
    if "unclear"   in c: return '<span class="b bu">? Unclear</span>'
    return '<span class="b bn">— N/P</span>'


def sources_html(s: str) -> str:
    urls = re.findall(r"https?://[^\s,<>\"'\]]+", s)
    if urls:
        return "<br>".join(f'<a href="{u}" target="_blank">🔗</a>' for u in urls[:3])
    # No URLs — show plain text (Stage 2 will verify)
    return f'<span style="color:#6b7280;font-size:10px">{s[:100]}</span>'


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
        tc         = "t1" if "1" in row["tier"] else "t2"
        hl         = "hl" if is_top else ""
        tcolor     = "#60a5fa" if "1" in row["tier"] else "#4b5563"
        tlabel     = "★ Tier 1" if "1" in row["tier"] else "Tier 2"
        bar        = min(100, int(score / 40 * 100))
        marker     = "🥇 " if is_top else ""
        sits       = row["situations"].replace(",", "<br>").replace(";", "<br>")
        ev         = row["key_evidence"].replace("•", "<br>•").replace("* ", "<br>• ")

        html_rows.append(f"""
        <tr class="{tc} {hl}">
          <td><b style="color:#f9fafb">{marker}{i+1}</b></td>
          <td><span style="font-weight:600;color:{tcolor}">{tlabel}</span></td>
          <td class="cn">{row['company']}</td>
          <td>{row['hq_country']}</td>
          <td>{row['industry']}</td>
          <td>{row['revenue']}</td>
          <td class="ev">{sits}</td>
          <td>{badge(row['classification'])}</td>
          <td><span class="sn">{score}/40</span>
              <div class="sbw"><div class="sb" style="width:{bar}%"></div></div>
              <small style="font-size:10px;color:#6b7280">{row['priority_score']}</small></td>
          <td><span style="font-size:10px;color:#6b7280">{freq}/{num_runs}</span></td>
          <td class="ev">{ev[:260]}</td>
          <td>{sources_html(row['sources'])}</td>
        </tr>""")

    return HTML_TEMPLATE.format(
        date=date_str, num_runs=num_runs,
        total=len(rows), tier1=tier1_count, tier2=tier2_count, top10=top_n,
        rows="\n".join(html_rows),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    prompt    = build_prompt()
    all_runs: list[list[dict]] = []
    raw_all:  list[str]        = []

    for i in range(NUM_RUNS):
        try:
            raw    = call_gemini(prompt, i)
            raw_all.append(raw)
            parsed = parse_markdown_table(raw)
            log.info(f"  → Parsed {len(parsed)} companies from run {i+1}")
            if len(parsed) == 0:
                log.warning(f"  → 0 companies. Raw preview:\n{raw[:1500]}")
            if parsed:
                all_runs.append(parsed)
        except Exception as e:
            log.warning(f"Run {i+1} failed: {e}")

        if (i + 1) % RUNS_BETWEEN_SLEEP == 0 and i < NUM_RUNS - 1:
            log.info(f"Sleeping {SLEEP_SECONDS}s …")
            time.sleep(SLEEP_SECONDS)

    if not all_runs:
        raise RuntimeError("No runs produced output. Check API key and response preview above.")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    (OUTPUT_DIR / f"stage1_raw_{ts}.json").write_text(
        json.dumps(raw_all, ensure_ascii=False, indent=2)
    )

    merged    = aggregate_runs(all_runs)
    html      = build_html(merged, NUM_RUNS)
    html_path = OUTPUT_DIR / f"stage1_chart_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Stage 1 chart → {html_path}")

    tier1_only    = [r for r in merged if "1" in r["tier"]]
    top_n         = min(TOP_N_FOR_STAGE2, len(tier1_only))
    top_companies = tier1_only[:top_n]

    log.info(f"Stage 2 input ({top_n} Tier 1 companies by score):")
    for r in top_companies:
        log.info(f"  → {r['company']:30s} | {r['_score_int']}/40")

    stage2_path = OUTPUT_DIR / "stage2_input.json"
    stage2_path.write_text(json.dumps(top_companies, ensure_ascii=False, indent=2))
    log.info(f"Stage 2 input saved → {stage2_path}")
    return html_path, stage2_path


if __name__ == "__main__":
    main()
