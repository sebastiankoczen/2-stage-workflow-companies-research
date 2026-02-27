"""
XIMPAX Intelligence Engine — Stage 1

Reads companies from config/companies.xlsx (the master list).
Each weekly run processes the next 100 companies: 10 runs × 10 companies each.
A state.json file tracks the offset so each week picks up where the previous left off.
After the full list is exhausted it wraps back to company #1.

No web search — Gemini knowledge-based scoring.
Stage 2 does the live research.
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
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
COMPANIES_PER_RUN  = 5      # companies evaluated per Gemini call
NUM_RUNS           = 5       # runs per week → 10 × 10 = 100 companies/week
RUNS_BETWEEN_SLEEP = 3
SLEEP_SECONDS      = 20
TOP_N_FOR_STAGE2   = 10
TODAY              = datetime.utcnow().strftime("%Y-%m-%d")

COMPANIES_FILE = CONFIG_DIR / "companies.xlsx"
STATE_FILE     = CONFIG_DIR / "state.json"


# ── Company list loader ────────────────────────────────────────────────────────
def load_company_list() -> list[dict]:
    """Read companies.xlsx and return list of company dicts."""
    if not COMPANIES_FILE.exists():
        raise FileNotFoundError(
            f"Company list not found at {COMPANIES_FILE}\n"
            "Please add companies.xlsx to the config/ directory.\n"
            "Use the companies_template.xlsx as a starting point."
        )
    df = pd.read_excel(COMPANIES_FILE, sheet_name="Companies", dtype=str)
    df = df.fillna("")

    # Normalise column names (strip whitespace, handle variations)
    df.columns = [c.strip() for c in df.columns]

    companies = []
    for _, row in df.iterrows():
        # Skip empty rows
        if not str(row.get("Company", "")).strip():
            continue
        companies.append({
            "no":                str(row.get("No", "")).strip(),
            "company":           str(row.get("Company", "")).strip(),
            "revenue_usd":       str(row.get("Revenue (USD approx.)", "")).strip(),
            "revenue_local":     str(row.get("Revenue (Local Currency)", "")).strip(),
            "fy":                str(row.get("FY", "")).strip(),
            "hq":                str(row.get("Headquarters", "")).strip(),
            "industry":          str(row.get("Industry", "")).strip(),
            "ownership":         str(row.get("Ownership", "")).strip(),
        })

    log.info(f"Loaded {len(companies)} companies from {COMPANIES_FILE.name}")
    return companies


# ── State management ───────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"next_offset": 0, "last_run_date": None, "last_run_companies": [], "total_processed_all_time": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    log.info(f"State saved → next_offset: {state['next_offset']}")


def get_this_weeks_companies(all_companies: list[dict]) -> tuple[list[dict], int, int]:
    """
    Returns (this_weeks_100, offset_start, next_offset).
    Wraps around if the list end is reached.
    """
    state    = load_state()
    total    = len(all_companies)
    offset   = state.get("next_offset", 0) % total

    end      = offset + (COMPANIES_PER_RUN * NUM_RUNS)  # = 100
    if end <= total:
        batch = all_companies[offset:end]
    else:
        # Wrap around
        batch = all_companies[offset:] + all_companies[:end - total]
        log.info(f"Wrap-around: {total - offset} from end + {end - total} from start")

    next_offset = end % total
    log.info(f"This week: companies #{offset+1}–{min(end, total)} "
             f"(offset {offset} → {next_offset}, total list: {total})")
    return batch, offset, next_offset


# ── Prompt builder ─────────────────────────────────────────────────────────────
def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_prompt(batch: list[dict], run_index: int) -> str:
    company_profile = load_file(CONFIG_DIR / "company_profile.txt")
    icp_blueprint   = load_file(CONFIG_DIR / "icp_blueprint.txt")
    instructions    = load_file(CONFIG_DIR / "instructions.txt")
    prompt_template = load_file(PROMPTS_DIR / "prompt_1.txt")

    # Format the 10 companies for this run as a clear list
    company_lines = []
    for i, c in enumerate(batch, 1):
        line = (
            f"  {i}. {c['company']}"
            f" | Revenue: {c['revenue_usd']}"
            f" | HQ: {c['hq']}"
            f" | Industry: {c['industry']}"
            f" | Ownership: {c['ownership']}"
        )
        if c["fy"]:
            line += f" | FY: {c['fy']}"
        company_lines.append(line)

    companies_block = "\n".join(company_lines)

    return f"""
You are a senior supply chain market intelligence analyst. Today is {TODAY}.
This is run {run_index + 1} of {NUM_RUNS}.

=== FIRM PROFILE ===
{company_profile}

=== ICP BLUEPRINT ===
{icp_blueprint}

=== SITUATION DETECTION FRAMEWORK ===
{instructions}

=== YOUR TASK ===
Evaluate EXACTLY these {COMPANIES_PER_RUN} specific companies. Do not add, remove, or replace any.
Score each company on all four supply chain situations using your knowledge of their
2024–2025 corporate events, earnings releases, restructurings, M&A activity, and
supply chain news.

COMPANIES TO EVALUATE (run {run_index + 1}):
{companies_block}

=== MANDATORY EVALUATION — ALL 4 SITUATIONS PER COMPANY ===
For EVERY company, explicitly score all four independently:

SITUATION 1 — RESOURCE CONSTRAINTS:
  Leadership churn (CPO/VP SC departed or replaced)? Explicit bandwidth / capacity mentions?
  High SC/Procurement vacancy volumes without urgency resolution?
  ERP / IBP / S&OP transformation programs stalled or delayed?
  Score: each confirmed signal → STRONG +2 or MEDIUM +1, cap 10

SITUATION 2 — MARGIN PRESSURE:
  EBITDA or gross margin decline reported as structural (not one-off)?
  Quantified savings / cost-out program with targets and timeline?
  Guidance downgrade due to input costs or inefficiencies?
  Plant closure, SKU rationalization, portfolio exit for margin recovery?
  Score: each confirmed signal → STRONG +2 or MEDIUM +1, cap 10

SITUATION 3 — SIGNIFICANT GROWTH:
  M&A activity requiring SC integration? New plant / DC / capacity announced?
  Revenue growth outpacing operational infrastructure?
  Geographic expansion into new markets requiring SC redesign?
  Score: each confirmed signal → STRONG +2 or MEDIUM +1, cap 10

SITUATION 4 — SUPPLY CHAIN DISRUPTION:
  Production shutdown, product recall, quality crisis with supply impact?
  Missed guidance explicitly due to supply issues?
  Supplier failure, force majeure, logistics crisis with material impact?
  Score: each confirmed signal → STRONG +2 or MEDIUM +1, cap 10

SCORING THRESHOLDS:
  STRONG +2 | MEDIUM +1 | Max 10 per situation | Total max 40
  CONFIRMED ≥ 7 pts | LIKELY 4–6 pts | UNCLEAR 2–3 pts | NOT PRESENT 0–1 pts
  TIER 1 = ICP match + at least 1 situation CONFIRMED or LIKELY
  TIER 2 = ICP match but all situations UNCLEAR or NOT PRESENT

=== TASK ===
{prompt_template}

=== OUTPUT RULES ===
- Output ONLY the markdown table — no preamble, no commentary, no explanations
- Every row must start AND end with |
- Output EXACTLY {COMPANIES_PER_RUN} rows, one per company listed above — no more, no less
- Use the exact company names from the list above
- Priority Score format: RC: X | MP: X | SG: X | SCD: X = XX/40
- Sources: cite publication type and approximate date (e.g. "Reuters Q4 2024 earnings")
  Stage 2 will verify URLs — do not guess specific URLs here
- If you have limited knowledge of a company, still evaluate based on what you know
  and mark uncertain signals as MEDIUM rather than omitting them
"""


# ── Gemini call ────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are a supply chain market intelligence analyst. You have detailed knowledge "
    "of 2024–2025 corporate events for Swiss and European industrial/pharma/FMCG "
    "companies: earnings releases, restructurings, M&A activity, management changes, "
    "capacity investments, supply chain disruptions. "
    "When asked to evaluate specific companies, always output all requested rows. "
    "Output ONLY the markdown table. Start with the | header row. "
    "Every row must start AND end with |. Output all rows. Do not truncate."
)


def call_gemini(prompt: str, run_index: int, batch: list[dict]) -> str:
    client       = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    company_names = ", ".join(c["company"] for c in batch[:3]) + "..."
    log.info(f"Run {run_index+1}/{NUM_RUNS} → evaluating: {company_names}")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,   # lower = more consistent scoring
            max_output_tokens=16000,
        ),
    )
    text = response.text or ""
    log.info(f"  → {len(text)} chars returned")
    if len(text) < 300:
        log.warning(f"  → Short response: {text[:300]}")
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
        if cells[0].strip().lower() in ("tier", "col a", "#", ""):
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


# ── Aggregation ────────────────────────────────────────────────────────────────
def normalise_name(name: str) -> str:
    n = name.strip().upper()
    n = re.sub(r"\b(AG|SA|GMBH|LTD|PLC|INC|LLC|NV|BV|SE|SPA|SAS|HOLDING|GROUP)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def aggregate_runs(all_runs: list[list[dict]]) -> list[dict]:
    seen: dict[str, dict] = {}

    for run_rows in all_runs:
        for row in run_rows:
            name  = normalise_name(row["company"])
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
    # Primary: Tier 1 first. Secondary: composite of score + frequency
    # (frequency matters when scores are 0 — seen 3×/week > seen 1×)
    merged.sort(key=lambda r: (
        0 if "1" in r["tier"] else 1,
        -(r["_score_int"] * 3 + r["_frequency"]),
        -r["_score_int"],
        -r["_frequency"],
    ))

    log.info(f"Aggregated: {len(merged)} unique companies this week")
    for r in merged[:15]:
        log.info(f"  {r['company']:35s} | {r['_score_int']:2d}/40 | {r['tier']} | seen {r['_frequency']}x")
    return merged


# ── HTML chart ─────────────────────────────────────────────────────────────────
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
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; background: #1f2937;
           border-radius: 8px; overflow: hidden; border: 1px solid #374151; }}
  thead {{ background: #0f172a; }}
  thead th {{ padding: 10px 8px; text-align: left; font-size: 11px; color: #94a3b8;
              text-transform: uppercase; letter-spacing: .5px; white-space: nowrap;
              border-bottom: 2px solid #2563eb; }}
  tbody tr {{ border-bottom: 1px solid #374151; }}
  tbody tr:hover {{ background: #273548; }}
  tbody td {{ padding: 9px 8px; vertical-align: top; color: #d1d5db; }}
  .t1 {{ border-left: 3px solid #2563eb; }}
  .t2 {{ border-left: 3px solid #4b5563; }}
  .bc {{ display:inline-block;padding:2px 7px;border-radius:12px;font-size:10px;font-weight:600;background:#14532d;color:#86efac; }}
  .bl {{ display:inline-block;padding:2px 7px;border-radius:12px;font-size:10px;font-weight:600;background:#1e3a5f;color:#93c5fd; }}
  .bu {{ display:inline-block;padding:2px 7px;border-radius:12px;font-size:10px;font-weight:600;background:#422006;color:#fcd34d; }}
  .bn {{ display:inline-block;padding:2px 7px;border-radius:12px;font-size:10px;font-weight:600;background:#1f2937;color:#6b7280;border:1px solid #374151; }}
  .sbw {{ width:100%;background:#374151;border-radius:4px;height:6px;margin-top:4px; }}
  .sb  {{ height:6px;border-radius:4px;background:linear-gradient(90deg,#2563eb,#7c3aed); }}
  .sn  {{ font-weight:700;font-size:13px;color:#60a5fa; }}
  a    {{ color:#60a5fa;text-decoration:none;font-size:11px; }}
  .ev  {{ font-size:11px;line-height:1.5;color:#9ca3af; }}
  .hl td {{ background:#1e3a5f !important; }}
  .cn  {{ color:#f9fafb;font-weight:600; }}
  .note {{ margin-top:14px;font-size:11px;color:#6b7280; }}
  .batch-info {{ background:#1e293b;border:1px solid #334155;border-radius:6px;padding:10px 16px;
                 margin-bottom:16px;font-size:12px;color:#94a3b8; }}
</style>
</head>
<body>
<h1>XIMPAX Market Intelligence — Stage 1 Report</h1>
<p class="sub">Generated: {date} | {num_runs} runs of {per_run} companies | 100 companies evaluated this week</p>
<div class="batch-info">
  📋 <b>This week's batch:</b> Companies #{batch_start}–#{batch_end} from master list
  &nbsp;|&nbsp; Next week starts at: #{next_offset_plus1}
  &nbsp;|&nbsp; Total processed all-time: {total_processed}
</div>
<div class="stats">
  <div class="sc"><div class="v">{scanned}</div><div class="l">Evaluated</div></div>
  <div class="sc"><div class="v">{tier1}</div><div class="l">Tier 1</div></div>
  <div class="sc"><div class="v">{top10}</div><div class="l">→ Stage 2</div></div>
  <div class="sc"><div class="v">{avg_score}</div><div class="l">Avg Score</div></div>
</div>
<table>
<thead>
  <tr>
    <th>#</th><th>Tier</th><th>Company</th><th>HQ</th><th>Industry</th>
    <th>Revenue</th><th>Situations</th><th>Classification</th>
    <th>Score</th><th>Key Evidence</th><th>Sources</th>
  </tr>
</thead>
<tbody>{rows}</tbody>
</table>
<p class="note">🥇 Highlighted = Top {top10} companies forwarded to Stage 2 for live research.</p>
</body>
</html>"""


def badge(cls: str) -> str:
    c = cls.lower()
    if "confirmed" in c: return '<span class="bc">✓ Confirmed</span>'
    if "likely"    in c: return '<span class="bl">~ Likely</span>'
    if "unclear"   in c: return '<span class="bu">? Unclear</span>'
    return '<span class="bn">— N/P</span>'


def build_html(rows: list[dict], num_runs: int, batch_meta: dict) -> str:
    date_str    = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tier1_rows  = [r for r in rows if "1" in r["tier"]]
    top_n       = min(TOP_N_FOR_STAGE2, len(tier1_rows))
    top_set     = set(id(r) for r in tier1_rows[:top_n])
    avg         = round(sum(r["_score_int"] for r in rows) / max(len(rows), 1), 1)

    html_rows = []
    for i, row in enumerate(rows):
        is_top = id(row) in top_set
        score  = row["_score_int"]
        tc_col = "#60a5fa" if "1" in row["tier"] else "#4b5563"
        tl     = "★ T1" if "1" in row["tier"] else "T2"
        bar    = min(100, int(score / 40 * 100))
        marker = "🥇 " if is_top else ""
        sits   = row["situations"].replace(",", "<br>").replace(";", "<br>")
        ev     = row["key_evidence"][:260].replace("•", "<br>•")
        src    = f'<span style="color:#6b7280;font-size:10px">{row["sources"][:80]}</span>'
        hl     = "hl" if is_top else ""
        tc     = "t1" if "1" in row["tier"] else "t2"

        html_rows.append(f"""
        <tr class="{tc} {hl}">
          <td><b style="color:#f9fafb">{marker}{i+1}</b></td>
          <td><span style="font-weight:600;color:{tc_col}">{tl}</span></td>
          <td class="cn">{row['company']}</td>
          <td>{row['hq_country']}</td>
          <td>{row['industry']}</td>
          <td>{row['revenue']}</td>
          <td class="ev">{sits}</td>
          <td>{badge(row['classification'])}</td>
          <td><span class="sn">{score}/40</span>
              <div class="sbw"><div class="sb" style="width:{bar}%"></div></div>
              <small style="font-size:10px;color:#6b7280">{row['priority_score'][:40]}</small></td>
          <td class="ev">{ev}</td>
          <td>{src}</td>
        </tr>""")

    return HTML_TEMPLATE.format(
        date=date_str,
        num_runs=num_runs,
        per_run=COMPANIES_PER_RUN,
        scanned=len(rows),
        tier1=len(tier1_rows),
        top10=top_n,
        avg_score=avg,
        batch_start=batch_meta["batch_start"],
        batch_end=batch_meta["batch_end"],
        next_offset_plus1=batch_meta["next_offset"] + 1,
        total_processed=batch_meta["total_processed"],
        rows="\n".join(html_rows),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load full company list from Excel
    all_companies = load_company_list()

    # Get this week's 100 companies and advance the pointer
    batch_100, offset_start, next_offset = get_this_weeks_companies(all_companies)

    log.info(f"Stage 1: evaluating {len(batch_100)} companies | "
             f"List positions #{offset_start+1}–#{min(offset_start+100, len(all_companies))}")

    all_runs: list[list[dict]] = []
    raw_all:  list[str]        = []

    for i in range(NUM_RUNS):
        run_batch = batch_100[i * COMPANIES_PER_RUN : (i + 1) * COMPANIES_PER_RUN]
        if not run_batch:
            log.warning(f"Run {i+1}: no companies — batch exhausted")
            break

        try:
            prompt = build_prompt(run_batch, i)
            raw    = call_gemini(prompt, i, run_batch)
            raw_all.append(raw)
            parsed = parse_markdown_table(raw)
            log.info(f"  → Parsed {len(parsed)} rows for run {i+1}")
            if len(parsed) == 0:
                log.warning(f"  → 0 rows. Raw start:\n{raw[:600]}")
            if parsed:
                all_runs.append(parsed)
        except Exception as e:
            log.warning(f"Run {i+1} failed: {e}")

        if (i + 1) % RUNS_BETWEEN_SLEEP == 0 and i < NUM_RUNS - 1:
            log.info(f"Sleeping {SLEEP_SECONDS}s …")
            time.sleep(SLEEP_SECONDS)

    if not all_runs:
        raise RuntimeError("No runs produced output.")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    (OUTPUT_DIR / f"stage1_raw_{ts}.json").write_text(
        json.dumps(raw_all, ensure_ascii=False, indent=2))

    merged = aggregate_runs(all_runs)

    # Update and save state AFTER successful run
    state = load_state()
    state["next_offset"]              = next_offset
    state["last_run_date"]            = TODAY
    state["last_run_companies"]       = [c["company"] for c in batch_100]
    state["total_processed_all_time"] = state.get("total_processed_all_time", 0) + len(batch_100)
    save_state(state)

    # Build HTML chart
    batch_meta = {
        "batch_start":     offset_start + 1,
        "batch_end":       offset_start + len(batch_100),
        "next_offset":     next_offset,
        "total_processed": state["total_processed_all_time"],
    }
    html      = build_html(merged, NUM_RUNS, batch_meta)
    html_path = OUTPUT_DIR / f"stage1_chart_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Stage 1 chart → {html_path}")

    # Select top N for Stage 2
    tier1_only    = [r for r in merged if "1" in r["tier"]]
    top_n         = min(TOP_N_FOR_STAGE2, len(tier1_only))
    top_companies = tier1_only[:top_n]

    log.info(f"Stage 2 input ({top_n} companies):")
    for r in top_companies:
        log.info(f"  → {r['company']:35s} | {r['_score_int']}/40")

    stage2_path = OUTPUT_DIR / "stage2_input.json"
    stage2_path.write_text(json.dumps(top_companies, ensure_ascii=False, indent=2))
    log.info(f"Stage 2 input saved → {stage2_path}")

    return html_path, stage2_path


if __name__ == "__main__":
    main()
