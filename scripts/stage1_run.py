"""
XIMPAX Intelligence Engine — Stage 1 (v2)

Architecture: one Gemini call per company WITH Google Search grounding.
Processes 20 companies per week, one at a time, with sleep between calls.
Top 5 by priority score → Stage 2 for full deep scan.

Why this is better than batching 10 companies per call:
  - Full model attention per company (vs 10-way split)
  - Live Google Search fills knowledge gaps Gemini training misses
  - Accurate scoring before Stage 2 so the right 5 get deep-scanned
"""

import os
import re
import json
import time
import logging
from datetime import datetime
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
COMPANIES_PER_WEEK = 20    # companies scanned per weekly run (one call each)
SLEEP_BETWEEN      = 15    # seconds between company calls
TOP_N_FOR_STAGE2   = 5     # top companies forwarded to Stage 2
TODAY              = datetime.utcnow().strftime("%Y-%m-%d")

COMPANIES_FILE = CONFIG_DIR / "companies.xlsx"
STATE_FILE     = CONFIG_DIR / "state.json"

SIT_KEYS  = ["RC", "MP", "SG", "SCD"]
SIT_NAMES = {
    "RC":  "Resource Constraints",
    "MP":  "Margin Pressure",
    "SG":  "Significant Growth",
    "SCD": "Supply Chain Disruption",
}


# ── Company list loader ────────────────────────────────────────────────────────
def load_company_list() -> list[dict]:
    if not COMPANIES_FILE.exists():
        raise FileNotFoundError(f"Company list not found at {COMPANIES_FILE}")
    df = pd.read_excel(COMPANIES_FILE, sheet_name="Companies", dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    companies = []
    for _, row in df.iterrows():
        if not str(row.get("Company", "")).strip():
            continue
        companies.append({
            "no":            str(row.get("No", "")).strip(),
            "company":       str(row.get("Company", "")).strip(),
            "revenue_usd":   str(row.get("Revenue (USD approx.)", "")).strip(),
            "revenue_local": str(row.get("Revenue (Local Currency)", "")).strip(),
            "fy":            str(row.get("FY", "")).strip(),
            "hq":            str(row.get("Headquarters", "")).strip(),
            "industry":      str(row.get("Industry", "")).strip(),
            "ownership":     str(row.get("Ownership", "")).strip(),
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
    return {"next_offset": 0, "last_run_date": None,
            "last_run_companies": [], "total_processed_all_time": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    log.info(f"State saved → next_offset: {state['next_offset']}")


def get_this_weeks_companies(all_companies: list[dict]) -> tuple[list[dict], int, int]:
    state  = load_state()
    total  = len(all_companies)
    offset = state.get("next_offset", 0) % total
    end    = offset + COMPANIES_PER_WEEK
    if end <= total:
        batch = all_companies[offset:end]
    else:
        batch = all_companies[offset:] + all_companies[:end - total]
        log.info(f"Wrap-around: {total - offset} from end + {end - total} from start")
    next_offset = end % total
    log.info(
        f"This week: companies #{offset+1}–{min(end, total)} "
        f"(offset {offset} → {next_offset}, total list: {total})"
    )
    return batch, offset, next_offset


# ── Prompt ─────────────────────────────────────────────────────────────────────
def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_scan_prompt(company: dict) -> str:
    company_profile = load_file(CONFIG_DIR / "company_profile.txt")
    icp_blueprint   = load_file(CONFIG_DIR / "icp_blueprint.txt")
    instructions    = load_file(CONFIG_DIR / "instructions.txt")
    year            = TODAY[:4]

    return f"""You are a senior supply chain market intelligence analyst. Today is {TODAY}.
Only use evidence from the last 12 months (after {TODAY[:7]}-01).

=== COMPANY TO SCAN ===
Company:   {company['company']}
HQ:        {company['hq']}
Industry:  {company['industry']}
Revenue:   {company['revenue_usd']} USD | {company['revenue_local']} local | FY {company['fy']}
Ownership: {company['ownership']}

=== XIMPAX FIRM PROFILE ===
{company_profile}

=== ICP BLUEPRINT ===
{icp_blueprint}

=== SITUATION SIGNAL DEFINITIONS ===
{instructions}

═══════════════════════════════════════
TASK: Run these searches, then score
═══════════════════════════════════════
1. "{company['company']} supply chain procurement staffing restructuring {year}"
2. "{company['company']} EBITDA margin cost savings program {year}"
3. "{company['company']} acquisition merger capacity expansion plant {year}"
4. "{company['company']} supply disruption production halt recall {year}"

Score all four situations using STRONG +2 / MEDIUM +1 signals, cap 10 per situation.

CONFIRMED >= 7pts | LIKELY 4-6pts | UNCLEAR 2-3pts | NOT PRESENT 0-1pts
TIER 1 = ICP match + at least one CONFIRMED or LIKELY
TIER 2 = ICP match but all UNCLEAR or NOT PRESENT
OUT    = does not match ICP (bank, utility, insurer, transport, real estate)

═══════════════════════════════════════
OUTPUT — use EXACTLY this format
═══════════════════════════════════════
COMPANY: {company['company']}
TIER: [1 or 2 or OUT]

RC: [CONFIRMED/LIKELY/UNCLEAR/NOT PRESENT] | [X] pts
RC_SIGNAL: [strongest signal found, or "none"]
RC_SOURCE: [publication and approx date, or "none"]

MP: [CONFIRMED/LIKELY/UNCLEAR/NOT PRESENT] | [X] pts
MP_SIGNAL: [strongest signal found, or "none"]
MP_SOURCE: [publication and approx date, or "none"]

SG: [CONFIRMED/LIKELY/UNCLEAR/NOT PRESENT] | [X] pts
SG_SIGNAL: [strongest signal found, or "none"]
SG_SOURCE: [publication and approx date, or "none"]

SCD: [CONFIRMED/LIKELY/UNCLEAR/NOT PRESENT] | [X] pts
SCD_SIGNAL: [strongest signal found, or "none"]
SCD_SOURCE: [publication and approx date, or "none"]

TOTAL: [sum]/40
SUMMARY: [2-3 sentences on the most relevant supply chain signals found]"""


# ── Gemini call ────────────────────────────────────────────────────────────────
def scan_company(company: dict, idx: int, total: int) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cname  = company["company"]
    log.info(f"[{idx+1}/{total}] Scanning: {cname}")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_scan_prompt(company),
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
            max_output_tokens=4000,
        ),
    )
    text = response.text or ""
    log.info(f"  → {len(text)} chars for {cname}")
    if len(text) < 150:
        log.warning(f"  → Very short response for {cname}: {text[:200]}")
    return text


# ── Parser ─────────────────────────────────────────────────────────────────────
def parse_score(line: str) -> int:
    m = re.search(r"\|\s*(\d+)\s*pts?", line, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\|\s*(\d+)", line)
    if m:
        return int(m.group(1))
    return 0


def parse_result(raw: str, company: dict) -> dict:
    def field(key: str) -> str:
        m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", raw, re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    tier_raw = field("TIER").upper()
    if "1" in tier_raw:
        tier = "Tier 1"
    elif "OUT" in tier_raw:
        tier = "Out of ICP"
    else:
        tier = "Tier 2"

    sit_results = {}
    total_score = 0

    for key in SIT_KEYS:
        line   = field(key)
        status = "NOT PRESENT"
        for s in ("CONFIRMED", "LIKELY", "UNCLEAR", "NOT PRESENT"):
            if s in line.upper():
                status = s
                break
        pts    = parse_score(line)
        signal = field(f"{key}_SIGNAL")
        source = field(f"{key}_SOURCE")
        total_score += pts
        sit_results[key] = {"status": status, "pts": pts,
                             "signal": signal, "source": source}

    # Override total if explicit TOTAL line present
    total_line = field("TOTAL")
    m = re.search(r"(\d+)\s*/\s*40", total_line)
    if m:
        total_score = int(m.group(1))

    summary = field("SUMMARY")

    # Aggregate for HTML / Stage 2
    sit_labels = [
        f"{k}: {sit_results[k]['status']} ({sit_results[k]['pts']}pts)"
        for k in SIT_KEYS
        if sit_results[k]["status"] != "NOT PRESENT"
    ]
    situations_str = " | ".join(sit_labels) if sit_labels else "No signals detected"

    priority_score = (
        f"RC:{sit_results['RC']['pts']} | MP:{sit_results['MP']['pts']} | "
        f"SG:{sit_results['SG']['pts']} | SCD:{sit_results['SCD']['pts']} = {total_score}/40"
    )

    evidence_parts = [
        f"{k}: {sit_results[k]['signal']}"
        for k in SIT_KEYS
        if sit_results[k]["signal"].lower() not in ("none", "")
    ]
    key_evidence = " • ".join(evidence_parts) or "No signals detected"

    sources_parts = [
        sit_results[k]["source"]
        for k in SIT_KEYS
        if sit_results[k]["source"].lower() not in ("none", "")
    ]
    sources_str = "; ".join(dict.fromkeys(sources_parts))

    max_status = "NOT PRESENT"
    for key in SIT_KEYS:
        s = sit_results[key]["status"]
        if s == "CONFIRMED":
            max_status = "CONFIRMED"; break
        if s == "LIKELY" and max_status != "CONFIRMED":
            max_status = "LIKELY"
        if s == "UNCLEAR" and max_status == "NOT PRESENT":
            max_status = "UNCLEAR"

    return {
        "company":        company["company"],
        "hq_country":     company["hq"],
        "industry":       company["industry"],
        "revenue":        f"Approx. {company['revenue_usd']}",
        "situations":     situations_str,
        "classification": max_status,
        "priority_score": priority_score,
        "key_evidence":   key_evidence,
        "sources":        sources_str,
        "summary":        summary,
        "tier":           tier,
        "_score_int":     total_score,
        "_frequency":     1,
        "_raw":           raw,
        "_situations":    sit_results,
    }


# ── HTML ───────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XIMPAX Stage 1 — {date}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#111827;color:#e5e7eb;padding:24px}}
h1{{font-size:22px;margin-bottom:4px;color:#f9fafb}}
.sub{{color:#9ca3af;font-size:13px;margin-bottom:20px}}
.stats{{display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap}}
.sc{{background:#1f2937;border-radius:8px;padding:12px 20px;border:1px solid #374151;min-width:130px}}
.sc .v{{font-size:28px;font-weight:700;color:#60a5fa}}
.sc .l{{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px}}
.batch-info{{background:#1e293b;border:1px solid #334155;border-radius:6px;
             padding:10px 16px;margin-bottom:16px;font-size:12px;color:#94a3b8}}
table{{width:100%;border-collapse:collapse;font-size:12px;background:#1f2937;
       border-radius:8px;overflow:hidden;border:1px solid #374151}}
thead{{background:#0f172a}}
thead th{{padding:10px 8px;text-align:left;font-size:11px;color:#94a3b8;
          text-transform:uppercase;letter-spacing:.5px;white-space:nowrap;
          border-bottom:2px solid #2563eb}}
tbody tr{{border-bottom:1px solid #374151}}
tbody tr:hover{{background:#273548}}
tbody td{{padding:9px 8px;vertical-align:top;color:#d1d5db}}
.t1{{border-left:3px solid #2563eb}}
.t2{{border-left:3px solid #4b5563}}
.tout{{border-left:3px solid #374151;opacity:.5}}
.bc{{display:inline-block;padding:2px 7px;border-radius:12px;font-size:10px;font-weight:600;background:#14532d;color:#86efac}}
.bl{{display:inline-block;padding:2px 7px;border-radius:12px;font-size:10px;font-weight:600;background:#1e3a5f;color:#93c5fd}}
.bu{{display:inline-block;padding:2px 7px;border-radius:12px;font-size:10px;font-weight:600;background:#422006;color:#fcd34d}}
.bn{{display:inline-block;padding:2px 7px;border-radius:12px;font-size:10px;font-weight:600;background:#1f2937;color:#6b7280;border:1px solid #374151}}
.sbw{{width:100%;background:#374151;border-radius:4px;height:6px;margin-top:4px}}
.sb{{height:6px;border-radius:4px;background:linear-gradient(90deg,#2563eb,#7c3aed)}}
.sn{{font-weight:700;font-size:13px;color:#60a5fa}}
.ev{{font-size:11px;line-height:1.6;color:#9ca3af}}
.hl td{{background:#1e3a5f !important}}
.cn{{color:#f9fafb;font-weight:600}}
.note{{margin-top:14px;font-size:11px;color:#6b7280}}
.chip{{display:inline-block;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600;margin:1px}}
.cC{{background:#14532d;color:#86efac}}
.cL{{background:#1e3a5f;color:#93c5fd}}
.cU{{background:#422006;color:#fcd34d}}
.cN{{background:#1f2937;color:#4b5563;border:1px solid #374151}}
</style>
</head>
<body>
<h1>XIMPAX Market Intelligence — Stage 1 Report</h1>
<p class="sub">Generated: {date} | {scanned} companies scanned individually with live search | 1 call per company</p>
<div class="batch-info">
  📋 <b>This week's batch:</b> Companies #{batch_start}–#{batch_end} from master list
  &nbsp;|&nbsp; Next week starts at: #{next_offset_plus1}
  &nbsp;|&nbsp; Total processed all-time: {total_processed}
</div>
<div class="stats">
  <div class="sc"><div class="v">{scanned}</div><div class="l">Scanned</div></div>
  <div class="sc"><div class="v">{tier1}</div><div class="l">Tier 1</div></div>
  <div class="sc"><div class="v">{top_n}</div><div class="l">→ Stage 2</div></div>
  <div class="sc"><div class="v">{avg_score}</div><div class="l">Avg Score</div></div>
</div>
<table>
<thead><tr>
  <th>#</th><th>Tier</th><th>Company</th><th>HQ</th><th>Industry</th>
  <th>Revenue</th><th>Situations</th><th>Score</th><th>Top Signal</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
<p class="note">🥇 Highlighted = Top {top_n} companies forwarded to Stage 2 for deep scan.</p>
</body></html>"""


def sit_chip(key: str, status: str, pts: int) -> str:
    cls = {"CONFIRMED": "C", "LIKELY": "L", "UNCLEAR": "U"}.get(status, "N")
    pts_str = f" {pts}p" if pts > 0 else ""
    return f'<span class="chip c{cls}">{key}{pts_str}</span>'


def badge(status: str) -> str:
    s = status.upper()
    if "CONFIRMED" in s: return '<span class="bc">✓ Confirmed</span>'
    if "LIKELY"    in s: return '<span class="bl">~ Likely</span>'
    if "UNCLEAR"   in s: return '<span class="bu">? Unclear</span>'
    return '<span class="bn">— N/P</span>'


def build_html(results: list[dict], batch_meta: dict) -> str:
    date_str   = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tier1_rows = [r for r in results if r["tier"] == "Tier 1"]
    top_n      = min(TOP_N_FOR_STAGE2, len(tier1_rows))
    top_set    = set(id(r) for r in tier1_rows[:top_n])
    avg        = round(sum(r["_score_int"] for r in results) / max(len(results), 1), 1)

    html_rows = []
    for i, row in enumerate(results):
        is_top = id(row) in top_set
        score  = row["_score_int"]
        bar    = min(100, int(score / 40 * 100))
        marker = "🥇 " if is_top else ""
        hl     = "hl" if is_top else ""
        tc     = {"Tier 1": "t1", "Tier 2": "t2"}.get(row["tier"], "tout")
        tl_map = {"Tier 1": '<span style="font-weight:600;color:#60a5fa">★ T1</span>',
                  "Tier 2": '<span style="font-weight:600;color:#4b5563">T2</span>'}
        tl     = tl_map.get(row["tier"], '<span style="color:#374151">OUT</span>')

        sits_html = "".join(
            sit_chip(k, row["_situations"][k]["status"], row["_situations"][k]["pts"])
            for k in SIT_KEYS
        )

        # Top signal: first non-none MP or SG signal, then any
        top_sig = ""
        for k in ["MP", "SG", "RC", "SCD"]:
            s = row["_situations"][k]["signal"]
            if s and s.lower() not in ("none", ""):
                src = row["_situations"][k]["source"]
                top_sig = f'<b style="color:#d1d5db">{k}:</b> {s[:140]}'
                if src and src.lower() != "none":
                    top_sig += f'<br><span style="color:#6b7280;font-size:10px">{src[:60]}</span>'
                break
        if not top_sig:
            top_sig = '<span style="color:#4b5563">No signals detected</span>'

        html_rows.append(f"""
        <tr class="{tc} {hl}">
          <td><b style="color:#f9fafb">{marker}{i+1}</b></td>
          <td>{tl}</td>
          <td class="cn">{row['company']}</td>
          <td style="font-size:11px;color:#9ca3af">{row['hq_country']}</td>
          <td style="font-size:11px;color:#9ca3af">{row['industry']}</td>
          <td style="font-size:11px;color:#9ca3af">{row['revenue']}</td>
          <td style="white-space:nowrap">{sits_html}</td>
          <td><span class="sn">{score}/40</span>
              <div class="sbw"><div class="sb" style="width:{bar}%"></div></div></td>
          <td class="ev">{top_sig}</td>
        </tr>""")

    return HTML_TEMPLATE.format(
        date=date_str, scanned=len(results), tier1=len(tier1_rows),
        top_n=top_n, avg_score=avg,
        batch_start=batch_meta["batch_start"], batch_end=batch_meta["batch_end"],
        next_offset_plus1=batch_meta["next_offset"] + 1,
        total_processed=batch_meta["total_processed"],
        rows="\n".join(html_rows),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    all_companies = load_company_list()
    batch, offset_start, next_offset = get_this_weeks_companies(all_companies)

    log.info(
        f"Stage 1: {len(batch)} companies | 1-by-1 with Google Search | "
        f"Positions #{offset_start+1}–#{offset_start+len(batch)}"
    )

    results: list[dict] = []
    raw_all: dict       = {}

    for idx, company in enumerate(batch):
        try:
            raw    = scan_company(company, idx, len(batch))
            raw_all[company["company"]] = raw
            result = parse_result(raw, company)
            results.append(result)
            log.info(
                f"  → {result['company']:35s} | {result['_score_int']:2d}/40 "
                f"| {result['tier']} | {result['classification']}"
            )
        except Exception as e:
            log.warning(f"[{idx+1}/{len(batch)}] Failed: {company['company']}: {e}")
            results.append({
                "company": company["company"], "hq_country": company["hq"],
                "industry": company["industry"],
                "revenue": f"Approx. {company['revenue_usd']}",
                "situations": "SCAN FAILED", "classification": "NOT PRESENT",
                "priority_score": "RC:0|MP:0|SG:0|SCD:0=0/40",
                "key_evidence": f"Error: {str(e)[:100]}",
                "sources": "", "summary": "", "tier": "Tier 2",
                "_score_int": 0, "_frequency": 1, "_raw": "",
                "_situations": {
                    k: {"status": "NOT PRESENT", "pts": 0,
                        "signal": "none", "source": "none"}
                    for k in SIT_KEYS
                },
            })

        if idx < len(batch) - 1:
            log.info(f"  Sleeping {SLEEP_BETWEEN}s …")
            time.sleep(SLEEP_BETWEEN)

    # Sort: Tier 1 first, then by score
    results.sort(key=lambda r: (
        0 if r["tier"] == "Tier 1" else (1 if r["tier"] == "Tier 2" else 2),
        -r["_score_int"],
    ))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    (OUTPUT_DIR / f"stage1_raw_{ts}.json").write_text(
        json.dumps(raw_all, ensure_ascii=False, indent=2)
    )

    state = load_state()
    state["next_offset"]              = next_offset
    state["last_run_date"]            = TODAY
    state["last_run_companies"]       = [c["company"] for c in batch]
    state["total_processed_all_time"] = state.get("total_processed_all_time", 0) + len(batch)
    save_state(state)

    log.info(f"Stage 1 complete: {len(results)} companies")
    for r in results:
        log.info(f"  {r['company']:35s} | {r['_score_int']:2d}/40 | {r['tier']}")

    batch_meta = {
        "batch_start":     offset_start + 1,
        "batch_end":       offset_start + len(batch),
        "next_offset":     next_offset,
        "total_processed": state["total_processed_all_time"],
    }
    html      = build_html(results, batch_meta)
    html_path = OUTPUT_DIR / f"stage1_chart_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Stage 1 chart → {html_path}")

    # Top N Tier 1 → Stage 2. If not enough Tier 1, fill with Tier 2.
    tier1_only    = [r for r in results if r["tier"] == "Tier 1"]
    top_companies = tier1_only[:TOP_N_FOR_STAGE2]
    if len(top_companies) < TOP_N_FOR_STAGE2:
        tier2_fill = [r for r in results if r["tier"] == "Tier 2"]
        top_companies += tier2_fill[:TOP_N_FOR_STAGE2 - len(top_companies)]

    log.info(f"Stage 2 input ({len(top_companies)} companies):")
    for r in top_companies:
        log.info(f"  → {r['company']:35s} | {r['_score_int']}/40")

    stage2_path = OUTPUT_DIR / "stage2_input.json"
    stage2_path.write_text(json.dumps(top_companies, ensure_ascii=False, indent=2))
    log.info(f"Stage 2 input saved → {stage2_path}")

    return html_path, stage2_path


if __name__ == "__main__":
    main()
