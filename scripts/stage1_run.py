"""
XIMPAX Intelligence Engine — Stage 1

Key design: SECTOR ROTATION across 10 runs to force company diversity.
Each run is assigned a specific industry pair so 10 runs × 20 companies
explore the full Swiss industrial landscape rather than repeating the
same 5 famous names every run.

No web search — Gemini knowledge-based. Stage 2 does live research.
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
NUM_RUNS           = 10
RUNS_BETWEEN_SLEEP = 3
SLEEP_SECONDS      = 20
TOP_N_FOR_STAGE2   = 10

REVENUE_MIN_B = 0.5
REVENUE_MAX_B = 15.0
TODAY         = datetime.utcnow().strftime("%Y-%m-%d")

# ── Sector rotation config ─────────────────────────────────────────────────────
# 10 run slots mapped to industry focus + seed companies to include
# This guarantees breadth across the Swiss industrial landscape
SECTOR_ROTATION = [
    # Run 0: Pharma & Life Sciences
    {
        "focus": "Pharmaceuticals, Biotechnology, CDMO, Life Sciences",
        "seed_companies": "Lonza, Bachem, Siegfried, Molecular Partners, Idorsia, Relief Therapeutics, Obseva, Polyphor, Evolent Health, Vifor Pharma",
        "exclude": "",
    },
    # Run 1: MedTech & Medical Devices
    {
        "focus": "Medical Devices, Dental, Diagnostics, MedTech",
        "seed_companies": "Straumann, Sonova, Ypsomed, Tecan, Hamilton Medical, Schiller, Medela, Sensirion, Skan Group, Coltene",
        "exclude": "",
    },
    # Run 2: Specialty Chemicals & Materials
    {
        "focus": "Specialty Chemicals, Coatings, Adhesives, Construction Chemicals, Plastics",
        "seed_companies": "Sika, Clariant, EMS-Chemie, Dätwyler, Bossard, Huber+Suhner, Gurit, Komax, Bucher Industries, List",
        "exclude": "",
    },
    # Run 3: Flavours, Fragrances & Consumer Ingredients
    {
        "focus": "Flavors, Fragrances, Cosmetics Ingredients, Personal Care, Consumer Chemicals",
        "seed_companies": "Givaudan, Firmenich (pre-merger), Ineos Styrolution Switzerland, Carbogen Amcis, Lonza Consumer Health, Salvona, Alessa, Lipoid, Roquette Suisse, Ashland Switzerland",
        "exclude": "",
    },
    # Run 4: Food & Beverage
    {
        "focus": "Food Processing, Dairy, Beverages, Nutrition, Food Ingredients",
        "seed_companies": "Emmi, Lindt & Sprüngli, Orior, Bell Food Group, Hügli, Bernrain, Kambly, Wander (Ovomaltine), Hero Group, Rivella",
        "exclude": "",
    },
    # Run 5: Industrial Manufacturing & Automation
    {
        "focus": "Industrial Machinery, Automation, Robotics, Precision Manufacturing, Tools",
        "seed_companies": "Georg Fischer, Bobst, Komax, Bystronic, Schindler (lifts), Kistler, Endress+Hauser, Tornos, Fritz Studer, Liebherr Switzerland",
        "exclude": "",
    },
    # Run 6: Packaging & Printing
    {
        "focus": "Packaging Machinery, Labels, Films, Printed Materials, Packaging Materials",
        "seed_companies": "Bobst, SIG Group, Schweitzer-Mauduit International Switzerland, Vetropack, Aluflexpack, UNIQA, Hug Engineering, Constantia Flexibles Switzerland, Perlen Packaging, Billerudkorsnäs Switzerland",
        "exclude": "",
    },
    # Run 7: Luxury, Watches & Consumer Goods
    {
        "focus": "Luxury Goods, Watches, Jewelry, Premium Consumer Brands",
        "seed_companies": "Richemont (divisions only under $15B revenue threshold), Swatch Group divisions, Audemars Piguet, Patek Philippe, IWC Schaffhausen, TAG Heuer, Chopard, Breitling, Jaeger-LeCoultre, Vacheron Constantin",
        "exclude": "",
    },
    # Run 8: Agribusiness, Crop Science & Animal Health
    {
        "focus": "Agribusiness, Crop Protection, Animal Health, Seeds, Fertilizers",
        "seed_companies": "Syngenta (pre-merger Swiss ops), Virbac Switzerland, Elanco Switzerland, Bayer CropScience Switzerland, CEVA Santé Animale Switzerland, Omya, Fenaco, Landi, Debrunner Acifer, Landor",
        "exclude": "",
    },
    # Run 9: Energy, Utilities & Logistics
    {
        "focus": "Energy, Utilities, Grid Technology, Logistics Services, Transportation",
        "seed_companies": "Alpiq, BKW, Axpo, Meyer Burger, Landis+Gyr, ABB Switzerland divisions, Helion Energy, Kuehne+Nagel Switzerland ops, Planzer Transport, Camion Transport",
        "exclude": "",
    },
]


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_prompt(run_index: int) -> str:
    company_profile  = load_file(CONFIG_DIR / "company_profile.txt")
    icp_blueprint    = load_file(CONFIG_DIR / "icp_blueprint.txt")
    instructions     = load_file(CONFIG_DIR / "instructions.txt")
    prompt_template  = load_file(PROMPTS_DIR / "prompt_1.txt")

    sector = SECTOR_ROTATION[run_index % len(SECTOR_ROTATION)]

    return f"""
You are a senior supply chain market intelligence analyst. Today is {TODAY}.
This is run {run_index + 1} of {NUM_RUNS}.

=== THIS RUN'S INDUSTRY FOCUS ===
You must focus EXCLUSIVELY on: {sector['focus']}
Seed companies to consider (you may include others in this sector too):
{sector['seed_companies']}

=== FIRM PROFILE ===
{company_profile}

=== ICP BLUEPRINT ===
{icp_blueprint}

=== SITUATION DETECTION FRAMEWORK ===
{instructions}

=== HARD CAMPAIGN FILTERS ===
Revenue: USD {REVENUE_MIN_B}B minimum — USD {REVENUE_MAX_B}B MAXIMUM (hard cap, strictly enforced)
Region: Switzerland or companies with strong Swiss manufacturing/SC presence
EXCLUDED (too large): Roche, Novartis, Nestlé, ABB (group), Zurich Insurance, Swiss Re,
  UBS, Credit Suisse, Richemont group (>$15B), Swatch Group (>$7B as group entity),
  Kühne+Nagel group (>$33B)
NOTE on conglomerates: if a division/subsidiary is in scope, include it as that entity
  (e.g. "ABB Robotics Switzerland" not "ABB")

=== MANDATORY EVALUATION — ALL 4 SITUATIONS ===
For EVERY company, explicitly score all four independently:

SITUATION 1 — RESOURCE CONSTRAINTS:
  Leadership churn (CPO/VP SC departed)? Explicit bandwidth/capacity mentions?
  High SC/Procurement vacancy volumes? ERP/IBP/S&OP programs stalled?
  Score: each confirmed signal → STRONG +2 or MEDIUM +1, cap 10

SITUATION 2 — MARGIN PRESSURE:
  EBITDA or gross margin decline reported as structural?
  Quantified savings/cost-out program with targets?
  Guidance downgrade due to costs? Plant closure / SKU rationalization?
  Score: each confirmed signal → STRONG +2 or MEDIUM +1, cap 10

SITUATION 3 — SIGNIFICANT GROWTH:
  M&A activity requiring SC integration? New plant/DC/capacity announced?
  Revenue growth outpacing operational infrastructure? Geographic expansion?
  Score: each confirmed signal → STRONG +2 or MEDIUM +1, cap 10

SITUATION 4 — SUPPLY CHAIN DISRUPTION:
  Production shutdown, recall, quality crisis? Missed guidance due to supply?
  Supplier failure, force majeure, logistics disruption with material impact?
  Score: each confirmed signal → STRONG +2 or MEDIUM +1, cap 10

SCORING:
  STRONG +2 | MEDIUM +1 | Max 10/situation | Total max 40
  CONFIRMED = 7-10 pts | LIKELY = 4-6 pts | UNCLEAR = 2-3 pts | NOT PRESENT = 0-1 pts
  TIER 1 = ICP match + ≥1 situation CONFIRMED or LIKELY
  TIER 2 = ICP match but all situations UNCLEAR or NOT PRESENT

=== TASK ===
{prompt_template}

=== OUTPUT RULES (CRITICAL) ===
- Output ONLY the markdown table — no preamble, no commentary
- Every row must start AND end with |
- Include EXACTLY 20 companies from this run's sector focus
- DO NOT repeat companies you already covered in other runs' typical picks
  (avoid defaulting to: Lonza, Givaudan, Sika, Straumann, Sonova unless
  they are specifically in this run's sector focus)
- Use your knowledge of 2024 annual results, Q3/Q4 2024 earnings, restructurings,
  M&A activity, management changes for these specific sector companies
- Priority Score format: RC: X | MP: X | SG: X | SCD: X = XX/40
- Sources: cite publication type only (Reuters Q4 2024, Annual Report 2024, etc.)
  Stage 2 will verify URLs — do not guess specific URLs here
- Be generous with scoring: if a CDMO has a new plant, that's SG LIKELY minimum
"""


# ── Gemini call ────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are a supply chain market intelligence analyst specialising in Swiss and "
    "European companies. You have detailed knowledge of 2024 corporate events for "
    "mid-market companies (not just large caps): earnings, restructurings, M&A, "
    "management changes, supply chain news, capacity investments. "
    "Output ONLY the markdown table. Start with the | header row. "
    "Every row must start AND end with |. Output all 20 rows. Do not truncate. "
    "Score companies generously based on what you know — do not leave scores at 0."
)


def call_gemini(prompt: str, run_index: int) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sector_name = SECTOR_ROTATION[run_index % len(SECTOR_ROTATION)]["focus"].split(",")[0]
    log.info(f"Run {run_index+1}/{NUM_RUNS} → {sector_name} sector …")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.6,   # higher = more diverse company choices
            max_output_tokens=16000,
        ),
    )
    text = response.text or ""
    log.info(f"  → {len(text)} chars returned")
    if len(text) < 500:
        log.warning(f"  → Very short response: {text[:300]}")
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
def normalise_name(name: str) -> str:
    n = name.strip().upper()
    n = re.sub(r"\b(AG|SA|GMBH|LTD|PLC|INC|LLC|NV|BV|SE|SPA|SAS|HOLDING|GROUP)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def aggregate_runs(all_runs: list[list[dict]]) -> list[dict]:
    seen: dict[str, dict] = {}

    for run_rows in all_runs:
        for row in run_rows:
            if not revenue_in_range(row["revenue"]):
                log.info(f"  ✗ Revenue filter: {row['company']} ({row['revenue']})")
                continue

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
    merged.sort(key=lambda r: (
        0 if "1" in r["tier"] else 1,
        -r["_score_int"],
        -r["_frequency"],
    ))

    log.info(f"Aggregated: {len(merged)} unique companies after revenue filter")
    for r in merged[:10]:
        log.info(f"  {r['company']:35s} | {r['_score_int']:2d}/40 | {r['tier']} | {r['_frequency']}x")
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
  .sector-badge {{ display:inline-block;padding:1px 6px;border-radius:4px;font-size:9px;
                   background:#1e293b;color:#64748b;border:1px solid #334155;margin-left:4px; }}
</style>
</head>
<body>
<h1>XIMPAX Market Intelligence — Stage 1 Report</h1>
<p class="sub">Generated: {date} | {num_runs} sector-rotated runs | {total} unique companies found</p>
<div class="stats">
  <div class="sc"><div class="v">{total}</div><div class="l">Unique Companies</div></div>
  <div class="sc"><div class="v">{sectors}</div><div class="l">Sectors Covered</div></div>
  <div class="sc"><div class="v">{tier1}</div><div class="l">Tier 1 Leads</div></div>
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
<p class="note">🥇 Highlighted = Top {top10} Tier 1 companies forwarded to Stage 2 for live Perplexity research.</p>
</body>
</html>"""


def badge(cls: str) -> str:
    c = cls.lower()
    if "confirmed" in c: return '<span class="bc">✓ Confirmed</span>'
    if "likely"    in c: return '<span class="bl">~ Likely</span>'
    if "unclear"   in c: return '<span class="bu">? Unclear</span>'
    return '<span class="bn">— N/P</span>'


def build_html(rows: list[dict], num_runs: int) -> str:
    date_str    = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tier1_count = sum(1 for r in rows if "1" in r["tier"])
    tier2_count = len(rows) - tier1_count
    tier1_rows  = [r for r in rows if "1" in r["tier"]]
    top_n       = min(TOP_N_FOR_STAGE2, len(tier1_rows))
    top_set     = set(id(r) for r in tier1_rows[:top_n])

    html_rows = []
    for i, row in enumerate(rows):
        is_top = id(row) in top_set
        score  = row["_score_int"]
        freq   = row["_frequency"]
        tc     = "t1" if "1" in row["tier"] else "t2"
        hl     = "hl" if is_top else ""
        tc_col = "#60a5fa" if "1" in row["tier"] else "#4b5563"
        tl     = "★ T1" if "1" in row["tier"] else "T2"
        bar    = min(100, int(score / 40 * 100))
        marker = "🥇 " if is_top else ""
        sits   = row["situations"].replace(",", "<br>").replace(";", "<br>")
        ev     = row["key_evidence"][:260].replace("•", "<br>•")

        src_html = f'<span style="color:#6b7280;font-size:10px">{row["sources"][:80]}</span>'

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
          <td><span style="font-size:10px;color:#6b7280">{freq}/{num_runs}</span></td>
          <td class="ev">{ev}</td>
          <td>{src_html}</td>
        </tr>""")

    return HTML_TEMPLATE.format(
        date=date_str, num_runs=num_runs,
        total=len(rows), sectors=min(num_runs, len(SECTOR_ROTATION)),
        tier1=tier1_count, top10=top_n,
        rows="\n".join(html_rows),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    all_runs: list[list[dict]] = []
    raw_all:  list[str]        = []

    for i in range(NUM_RUNS):
        try:
            prompt = build_prompt(i)
            raw    = call_gemini(prompt, i)
            raw_all.append(raw)
            parsed = parse_markdown_table(raw)
            log.info(f"  → Parsed {len(parsed)} companies from run {i+1}")
            if len(parsed) == 0:
                log.warning(f"  → 0 companies. Raw:\n{raw[:800]}")
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
        log.info(f"  → {r['company']:35s} | {r['_score_int']}/40")

    stage2_path = OUTPUT_DIR / "stage2_input.json"
    stage2_path.write_text(json.dumps(top_companies, ensure_ascii=False, indent=2))
    log.info(f"Stage 2 input saved → {stage2_path}")
    return html_path, stage2_path


if __name__ == "__main__":
    main()
