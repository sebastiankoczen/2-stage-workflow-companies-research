"""
XIMPAX Intelligence Engine — Stage 2
Two-call architecture per company:
  Call A → Gemini + Google Search → prose research with real grounding URLs
  Call B → Gemini (no tools) → formats research into 4-column signal table

HTML report includes visual quality flags:
  🔴 Stale evidence (>12 months old)
  🟡 Forbidden source (company website / IR page)
  ✅ Clean signal
"""

import os
import re
import json
import time
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
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

GEMINI_MODEL  = "gemini-2.0-flash"
SLEEP_BETWEEN = 12   # seconds between companies

CUTOFF_DATE  = datetime.utcnow() - timedelta(days=365)
CUTOFF_STR   = CUTOFF_DATE.strftime("%Y-%m-%d")
TODAY        = datetime.utcnow().strftime("%Y-%m-%d")

# Sources that are explicitly forbidden per the framework
FORBIDDEN_SOURCE_PATTERNS = [
    r"\bcompany\s+website\b", r"\bcompany\s+site\b", r"\bIR\s+page\b",
    r"\binvestor\s+relations\b", r"\bpress\s+release\b",
    r"\bvendor\s+blog\b", r"\bswot\b",
    # Careers / jobs pages = company's own domain
    r"\bcareers?\b", r"\bjobs?\s+page\b", r"\bnewsroom\b",
    # Explicitly forbidden AI/SWOT analysis aggregator sites
    r"portersfiveforce", r"comparably", r"craft\.co", r"macroaxis",
    r"stockanalysis\.com", r"wisesheets", r"marketbeat", r"simplywall",
]

# When the model labels a source as just the company name (or company name + suffix)
# it means it used the company's own website / press room.
FORBIDDEN_SOURCE_LABELS = {
    "lonza", "givaudan", "novartis", "roche", "nestle", "abb", "sika",
    "sonova", "straumann", "georg fischer", "lindt", "emmi", "dätwyler",
    "tecan", "bossard", "orior", "huber", "bucher", "bobst",
}
# Suffixes that combined with a company name = company-owned page
COMPANY_PAGE_SUFFIXES = {
    "careers", "career", "jobs", "newsroom", "press", "news",
    "investor", "investors", "ir", "annual report", "sustainability report",
}


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_stage2_input() -> list[dict]:
    path = OUTPUT_DIR / "stage2_input.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 input not found at {path}. Run stage1_run.py first.")
    data = json.loads(path.read_text())
    data.sort(key=lambda r: r.get("_score_int", 0), reverse=True)
    log.info(f"Stage 2 input: {len(data)} companies (sorted by score):")
    for r in data:
        log.info(f"  → {r['company']:30s} | {r.get('_score_int', '?')}/40")
    return data


# ── CALL A: Research prompt ────────────────────────────────────────────────────
def build_research_prompt(company: dict) -> str:
    instructions  = load_file(CONFIG_DIR / "instructions.txt")
    icp_blueprint = load_file(CONFIG_DIR / "icp_blueprint.txt")

    return f"""
You are a senior supply chain market intelligence analyst. Today is {TODAY}.

Deep-scan this company for supply chain execution signals:
Company:  {company['company']}
Country:  {company['hq_country']}
Industry: {company['industry']}
Revenue:  {company['revenue']}
Stage 1 preliminary findings: {company.get('situations', 'unknown')} | Score: {company.get('priority_score', 'unknown')}

══════════════════════════════════════════════════════
MANDATORY DATE RULE — ZERO EXCEPTIONS
══════════════════════════════════════════════════════
The cutoff date is {CUTOFF_STR}. Every single piece of evidence MUST be
dated AFTER {CUTOFF_STR}. If you find an article and its publication date
is before {CUTOFF_STR}, you MUST discard it and search for a newer source.
Do NOT use any source older than 12 months. This is non-negotiable.

══════════════════════════════════════════════════════
MANDATORY SOURCE RULES — ZERO EXCEPTIONS  
══════════════════════════════════════════════════════
ALLOWED sources (use these):
  Tier 1: Reuters, Bloomberg, Financial Times, regulatory filings,
          earnings call transcripts (Seeking Alpha, Motley Fool, etc.)
  Tier 2: Trade publications (Pharma Tech, Chemical Week, etc.),
          Swissquote, NZZ, Neue Zürcher Zeitung, Swiss business press
  Tier 3: Reputable business news (Forbes, WSJ, etc.)

FORBIDDEN — do NOT use any of these:
  ✗ {company['company']} company website, press releases, newsroom, or careers page
  ✗ Any company's own investor relations page
  ✗ LinkedIn job posts (only acceptable for Organizational signals,
    and ONLY if the posting itself is dated after {CUTOFF_STR})
  ✗ Generic industry blogs or vendor content
  ✗ ZipRecruiter, Indeed, or other job boards for non-job-specific signals
  ✗ SWOT or AI analysis aggregator sites: PortersFiveForce.com, Comparably,
    Craft.co, Macroaxis, StockAnalysis, WiseSheets, MarketBeat, SimplyWallSt
  ✗ Generic industry articles that do NOT specifically name {company['company']}
    (e.g. "supply chain professionals face staffing challenges in 2024" is NOT
    evidence about {company['company']} — the article must name this company)
  ✗ Generic company taglines or marketing language
    (e.g. "world leader in X with lean manufacturing" is a company description,
    NOT evidence of a supply chain situation — it must be an analyst or journalist
    reporting a specific operational fact)

STRICT COUNTING RULES — READ CAREFULLY:
  ✗ The same article / URL may only be counted for ONE signal, not multiple
  ✗ If you use Reuters article X for one signal, you cannot use it again
  ✗ Maximum 2 signals from any single source publication per situation
  ✗ Do NOT count a signal without a real confirmed URL — write "no evidence found"

══════════════════════════════════════════════════════
SEARCH QUERIES — RUN ALL OF THESE
══════════════════════════════════════════════════════
Run each search and evaluate results before scoring:

SITUATION 1 — RESOURCE CONSTRAINTS:
  Search: "{company['company']} supply chain procurement staffing shortage capacity gap {TODAY[:4]}"
  Search: "{company['company']} CPO supply chain director departure vacancy hire {TODAY[:4]}"
  Search: "{company['company']} ERP SAP S4 IBP S&OP transformation delayed resource {TODAY[:4]}"

SITUATION 2 — MARGIN PRESSURE:
  Search: "{company['company']} EBITDA gross margin decline cost reduction restructuring {TODAY[:4]}"
  Search: "{company['company']} profit warning savings program procurement efficiency {TODAY[:4]}"
  Search: "{company['company']} layoffs plant closure footprint optimization {TODAY[:4]}"

SITUATION 3 — SIGNIFICANT GROWTH:
  Search: "{company['company']} acquisition merger plant expansion capacity investment {TODAY[:4]}"
  Search: "{company['company']} revenue growth supply chain scaling backlog {TODAY[:4]}"

SITUATION 4 — SUPPLY CHAIN DISRUPTION:
  Search: "{company['company']} supply disruption shortage production halt recall {TODAY[:4]}"
  Search: "{company['company']} supplier failure logistics disruption quality crisis {TODAY[:4]}"

══════════════════════════════════════════════════════
SCORING RULES
══════════════════════════════════════════════════════
STRONG signal = +2 pts | MEDIUM signal = +1 pt | Cap per situation = 10 pts
CONFIRMED ≥ 10 pts | LIKELY 6–9 pts | UNCLEAR 3–5 pts | NOT PRESENT 0–2 pts

=== SITUATION SIGNAL DEFINITIONS ===
{instructions}

=== ICP BLUEPRINT ===
{icp_blueprint}

══════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════
Write one block per situation:

SITUATION 1 — RESOURCE CONSTRAINTS: [CONFIRMED/LIKELY/UNCLEAR/NOT PRESENT] ([X] pts)
Signal: [exact signal name] | Strength: STRONG/MEDIUM | Weight: +2/+1
Quote: "[verbatim quote, max 25 words, must name {company['company']} explicitly]"
Date: YYYY-MM-DD (must be after {CUTOFF_STR}) | Source: [Publication name] | URL: [full https:// URL]
[repeat for each signal, or write "No evidence found within the last 12 months from allowed sources"]

SITUATION 2 — MARGIN PRESSURE: [classification] ([X] pts)
[same format]

SITUATION 3 — SIGNIFICANT GROWTH: [classification] ([X] pts)
[same format]

SITUATION 4 — SUPPLY CHAIN DISRUPTION: [classification] ([X] pts)
[same format]

TOTAL SCORE: RC:[n] | MP:[n] | SG:[n] | SCD:[n] = [total]/40
"""


# ── CALL B: Format prompt ──────────────────────────────────────────────────────
def build_format_prompt(company: dict, research_text: str) -> str:
    prompt_template = load_file(PROMPTS_DIR / "prompt_2.txt")
    return f"""
You are a data formatting engine. Convert the research findings into the required
markdown table. Do NOT do additional research. Do NOT change scores, quotes, or URLs.
Just reformat exactly what is in the research findings below.

=== RESEARCH FINDINGS ===
{research_text}

=== TABLE FORMAT REQUIRED ===
{prompt_template}

CRITICAL RULES:
1. Output ONLY the markdown table — no text before or after
2. Every row must start with | and end with |
3. Exactly 4 columns: Situation Status | Detected Signal | Evidence & Quote | Source & URL
4. One row per signal detected
5. In the Source & URL column: use ONLY URLs that appear verbatim in the research above
   — never invent, guess, abbreviate, or modify any URL
6. Include the publication date in the Evidence column: "(Published: YYYY-MM-DD)"
7. If research shows "No evidence found" for a situation, write one row with "No signals detected"
8. Signal weight labels MUST be exactly one of these two — nothing else:
   "STRONG +2"  (for strong signals worth 2 points)
   "MEDIUM +1"  (for medium signals worth 1 point)
   NEVER write "MEDIUM +2" — that is a labeling error. Medium is always +1.
9. The source label must be the publication name (Reuters, Bloomberg, FT, Swissquote etc.)
   NOT the article headline. If you only know the article title, use the domain name.
"""


# ── Gemini calls ───────────────────────────────────────────────────────────────
def gemini_research(prompt: str, company_name: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    log.info(f"  → Call A: researching {company_name} with Google Search …")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
            max_output_tokens=16000,
        ),
    )
    text = response.text or ""
    log.info(f"  → Call A: {len(text)} chars returned")
    if len(text) < 300:
        log.warning(f"  → Very short Call A response: {text[:300]}")
    return text


def gemini_format(research_text: str, company: dict) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    log.info(f"  → Call B: formatting {company['company']} into table …")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_format_prompt(company, research_text),
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a data formatting engine. Output ONLY the markdown table. "
                "Start with the | character of the header row. "
                "Every row must start and end with |. "
                "No text before or after the table. "
                "Never invent or modify URLs — only use URLs present in the input. "
                "Always include the publication date in the Evidence column."
            ),
            temperature=0.0,
            max_output_tokens=8000,
        ),
    )
    text = response.text or ""
    log.info(f"  → Call B: {len(text)} chars returned")
    return text


# ── Table parser ───────────────────────────────────────────────────────────────
def parse_stage2_table(raw: str) -> list[dict]:
    raw        = re.sub(r"```[a-z]*", "", raw)
    lines      = [l.strip() for l in raw.splitlines()]
    pipe_lines = [l for l in lines if l.startswith("|")]

    if not pipe_lines:
        log.warning("No pipe-delimited lines found in Stage 2 response.")
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
        if len(cells) < 4:
            continue
        while len(cells) < 4:
            cells.append("")

        rows.append({
            "situation_status": cells[0],
            "detected_signal":  cells[1],
            "evidence":         cells[2],
            "source_url":       cells[3],
        })
    return rows


# ── Signal quality validation ──────────────────────────────────────────────────
def parse_evidence_date(evidence: str) -> datetime | None:
    """Extract date from evidence string. Handles Published: YYYY-MM-DD format."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", evidence)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass
    return None


def classify_source(source_str: str) -> str:
    """
    Returns: 'forbidden' | 'clean'
    Covers: company websites, careers pages, IR pages, SWOT/AI analysis sites,
    article-headline-as-source-label (ambiguous).
    """
    s = source_str.lower().strip()

    # 1. Direct pattern matches (company site, press release, SWOT sites, etc.)
    for pat in FORBIDDEN_SOURCE_PATTERNS:
        if re.search(pat, s, re.IGNORECASE):
            return "forbidden"

    # 2. Company name alone OR company name + page suffix
    #    e.g. "Lonza" | "Lonza Careers" | "Sika Newsroom" | "Givaudan Press"
    for label in FORBIDDEN_SOURCE_LABELS:
        escaped = re.escape(label)
        # Bare company name (possibly with page ref in parens)
        if re.match(rf"^{escaped}(\s*[\(\[]|$)", s):
            return "forbidden"
        # Company name + forbidden suffix
        for suffix in COMPANY_PAGE_SUFFIXES:
            if re.match(rf"^{escaped}\s+{re.escape(suffix)}", s):
                return "forbidden"

    # 3. Source label looks like an article headline rather than a publication name
    #    Heuristic: >8 words, or contains "announces|agrees|plans|reports|falls|rises"
    #    These are unreliable because the publication can't be independently identified.
    words = s.split()
    if len(words) > 8:
        return "forbidden"
    headline_verbs = ["announces", "agreed", "plans", "reported", "falls", "rises",
                      "expands", "acquires", "closes", "layoffs", "restructur"]
    if any(v in s for v in headline_verbs):
        return "forbidden"

    return "clean"


def signal_quality(row: dict) -> dict:
    """
    Returns a dict with quality flags for a signal row:
      stale: bool  (evidence date before cutoff)
      forbidden: bool  (source is a company website)
      quality: 'clean' | 'stale' | 'forbidden' | 'stale+forbidden'
    """
    ev_date   = parse_evidence_date(row.get("evidence", ""))
    src_class = classify_source(row.get("source_url", ""))

    stale     = ev_date is not None and ev_date < CUTOFF_DATE
    forbidden = src_class == "forbidden"

    if stale and forbidden:
        quality = "stale+forbidden"
    elif stale:
        quality = "stale"
    elif forbidden:
        quality = "forbidden"
    else:
        quality = "clean"

    return {"stale": stale, "forbidden": forbidden, "quality": quality}


# ── Score recalculation ────────────────────────────────────────────────────────
def recalculate_score(rows: list[dict]) -> int:
    """Sum STRONG(+2) MEDIUM(+1) per situation, cap 10, return total."""
    sit_scores: dict[str, int] = {
        "resource constraints": 0,
        "margin pressure": 0,
        "significant growth": 0,
        "supply chain disruption": 0,
    }
    current = ""
    for row in rows:
        if row["situation_status"].strip():
            current = row["situation_status"].lower()
        sig = row.get("detected_signal", "").lower()
        if "no signals" in sig or not sig:
            continue
        # Only count clean signals in the score
        q = signal_quality(row)
        if q["quality"] != "clean":
            continue
        w = 2 if "strong" in sig else (1 if "medium" in sig else 0)
        for k in sit_scores:
            if k in current:
                sit_scores[k] = min(10, sit_scores[k] + w)
                break
    return sum(sit_scores.values())


# ── HTML report builder ────────────────────────────────────────────────────────
SITUATION_STYLES = {
    "confirmed":   ("#86efac", "#14532d", "✅ CONFIRMED"),
    "likely":      ("#93c5fd", "#1e3a5f", "🔵 LIKELY"),
    "unclear":     ("#fcd34d", "#422006", "⚠️ UNCLEAR"),
    "not present": ("#9ca3af", "#1f2937", "⬜ NOT PRESENT"),
}

def situation_style(s: str):
    t = s.lower()
    for k, v in SITUATION_STYLES.items():
        if k in t:
            return v
    return ("#d1d5db", "#1f2937", s[:40])


QUALITY_BADGES = {
    "clean":          ("", ""),
    "stale":          (' <span style="background:#7f1d1d;color:#fca5a5;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:700">⏰ STALE</span>',
                       "border-left:3px solid #ef4444;"),
    "forbidden":      (' <span style="background:#78350f;color:#fcd34d;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:700">🚫 EXCL.SOURCE</span>',
                       "border-left:3px solid #f59e0b;"),
    "stale+forbidden":(' <span style="background:#7f1d1d;color:#fca5a5;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:700">⏰ STALE</span>'
                       ' <span style="background:#78350f;color:#fcd34d;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:700">🚫 EXCL.SOURCE</span>',
                       "border-left:3px solid #ef4444;opacity:0.7;"),
}


def make_source_link(src: str, quality: str) -> str:
    urls  = re.findall(r"https?://[^\s<>\"'\]]+", src)
    label = re.sub(r"https?://[^\s<>\"'\]]+", "", src).strip(" —-–[]")
    badge, _ = QUALITY_BADGES.get(quality, ("", ""))

    if urls:
        link = f'<a href="{urls[0]}" target="_blank" style="color:#60a5fa">{label or "🔗 Source"}</a>'
    elif src.strip() and src.strip() not in ("-", "—"):
        link = f'<span style="color:#6b7280;font-size:10px">{src[:80]}</span>'
    else:
        link = ""
    return link + badge


def build_company_section(company: dict, rows: list[dict], rank: int) -> str:
    verified_score = recalculate_score(rows)
    raw_score      = company.get("_score_int", 0)
    bar_pct        = min(100, int(verified_score / 40 * 100))
    dom_color      = "#16a34a" if any("confirmed" in r["situation_status"].lower() for r in rows) else "#2563eb"

    # Count clean vs flagged signals for the summary line
    total_sig = sum(1 for r in rows if "no signals" not in r.get("detected_signal", "").lower() and r.get("detected_signal", ""))
    clean_sig = sum(1 for r in rows if signal_quality(r)["quality"] == "clean" and "no signals" not in r.get("detected_signal", "").lower())
    flag_sig  = total_sig - clean_sig

    quality_summary = (
        f'<span style="font-size:11px;color:#6b7280;margin-top:4px;display:block">'
        f'✅ {clean_sig} clean signals'
        + (f' &nbsp;|&nbsp; <span style="color:#fca5a5">⚠️ {flag_sig} flagged (stale/excluded source)</span>' if flag_sig else "")
        + '</span>'
    )

    # Group by situation
    groups: dict[str, list] = {}
    cur = ""
    for row in rows:
        if row["situation_status"].strip():
            cur = row["situation_status"]
        groups.setdefault(cur, []).append(row)

    sig_rows_html = []
    for sit, sig_rows in groups.items():
        tc, bg, label = situation_style(sit)
        first = True
        for r in sig_rows:
            q     = signal_quality(r)
            _, row_style = QUALITY_BADGES.get(q["quality"], ("", ""))

            sit_td = (
                f'<td style="background:{bg};color:{tc};font-weight:700;font-size:11px;'
                f'white-space:nowrap;padding:10px 8px;border-right:1px solid #374151">'
                f'{label}<br><small style="font-size:9px;font-weight:400;opacity:.8">{sit}</small></td>'
            ) if first else '<td style="background:#111827;border-right:1px solid #374151"></td>'
            first = False

            sig_rows_html.append(f"""
            <tr style="border-bottom:1px solid #374151;{row_style}">
              {sit_td}
              <td style="padding:10px 8px;font-size:12px;color:#d1d5db;border-right:1px solid #374151">{r['detected_signal']}</td>
              <td style="padding:10px 8px;font-size:11px;font-style:italic;color:#9ca3af;border-right:1px solid #374151">{r['evidence']}</td>
              <td style="padding:10px 8px;font-size:11px">{make_source_link(r['source_url'], q['quality'])}</td>
            </tr>""")

    no_sig = ('<tr><td colspan="4" style="padding:14px;color:#6b7280;text-align:center;'
              'font-style:italic">No signals detected within allowed sources and date range</td></tr>')

    return f"""
    <div style="background:#1f2937;border-radius:10px;margin-bottom:24px;overflow:hidden;border:1px solid #374151">
      <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-left:4px solid {dom_color};
                  padding:16px 20px;display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">#{rank} Priority Lead</div>
          <h2 style="font-size:18px;margin:0;color:#f9fafb;font-weight:700">{company['company']}</h2>
          <p style="font-size:12px;color:#9ca3af;margin:5px 0 2px">{company['industry']} &nbsp;·&nbsp; {company['hq_country']} &nbsp;·&nbsp; {company['revenue']}</p>
          {quality_summary}
        </div>
        <div style="text-align:right;min-width:140px">
          <div style="font-size:30px;font-weight:800;color:#60a5fa">{verified_score}<small style="font-size:14px;color:#6b7280">/40</small></div>
          <div style="background:rgba(255,255,255,.1);border-radius:4px;height:6px;width:110px;margin:6px 0 2px;margin-left:auto">
            <div style="background:linear-gradient(90deg,#2563eb,#7c3aed);height:6px;border-radius:4px;width:{bar_pct}%"></div>
          </div>
          <small style="font-size:10px;color:#6b7280">Verified score (clean signals only)</small>
          {f'<small style="font-size:10px;color:#6b7280;display:block">Stage 1 estimate: {raw_score}/40</small>' if raw_score != verified_score else ""}
        </div>
      </div>
      <div style="background:#1a2535;padding:8px 16px;border-bottom:1px solid #374151;font-size:10px;color:#6b7280">
        Legend: &nbsp;
        <span style="background:#7f1d1d;color:#fca5a5;padding:1px 5px;border-radius:4px;font-weight:700">⏰ STALE</span> = evidence older than {CUTOFF_STR} &nbsp;|&nbsp;
        <span style="background:#78350f;color:#fcd34d;padding:1px 5px;border-radius:4px;font-weight:700">🚫 EXCL.SOURCE</span> = company website / IR page (excluded by framework rules)
      </div>
      <table style="width:100%;border-collapse:collapse;font-family:Arial,sans-serif">
        <thead>
          <tr style="background:#111827;border-bottom:2px solid #374151">
            <th style="padding:8px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase;width:155px;border-right:1px solid #374151">Situation</th>
            <th style="padding:8px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase;border-right:1px solid #374151">Detected Signal</th>
            <th style="padding:8px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase;border-right:1px solid #374151">Evidence & Quote</th>
            <th style="padding:8px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase">Source</th>
          </tr>
        </thead>
        <tbody>{"".join(sig_rows_html) if sig_rows_html else no_sig}</tbody>
      </table>
    </div>"""


REPORT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>XIMPAX Deep Scan — {date}</title></head>
<body style="margin:0;padding:0;background:#0f172a;font-family:Arial,sans-serif;color:#e5e7eb">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;padding:24px 0">
<tr><td>
<table width="700" align="center" cellpadding="0" cellspacing="0" style="max-width:700px;margin:0 auto">

  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:12px 12px 0 0;
                 padding:28px 32px;border-bottom:3px solid #2563eb">
    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px">XIMPAX Intelligence Engine</div>
    <h1 style="margin:0;font-size:22px;color:#f9fafb;font-weight:800">🎯 Weekly Deep Scan Report</h1>
    <p style="margin:8px 0 0;color:#9ca3af;font-size:13px">Stage 2 · {date} · {num_companies} Companies · Evidence cutoff: {cutoff}</p>
  </td></tr>

  <tr><td style="background:#1f2937;padding:16px 32px;border-bottom:1px solid #374151">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td style="text-align:center;padding:8px 0;border-right:1px solid #374151">
        <div style="font-size:26px;font-weight:800;color:#60a5fa">{num_companies}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Scanned</div>
      </td>
      <td style="text-align:center;padding:8px 0;border-right:1px solid #374151">
        <div style="font-size:26px;font-weight:800;color:#86efac">{confirmed}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Confirmed</div>
      </td>
      <td style="text-align:center;padding:8px 0;border-right:1px solid #374151">
        <div style="font-size:26px;font-weight:800;color:#93c5fd">{likely}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Likely</div>
      </td>
      <td style="text-align:center;padding:8px 0">
        <div style="font-size:26px;font-weight:800;color:#fcd34d">{clean_signals}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Clean Signals</div>
      </td>
    </tr></table>
  </td></tr>

  <tr><td style="background:#111827;padding:20px 24px">{company_sections}</td></tr>

  <tr><td style="background:#0f172a;border-radius:0 0 12px 12px;padding:16px 32px;
                 border-top:1px solid #374151;text-align:center;color:#4b5563;font-size:11px">
    XIMPAX Intelligence Engine · Automated Weekly Report · Gemini AI + Google Search<br>
    Evidence window: {cutoff} → {date} · Excluded sources: company websites, IR pages, vendor blogs
  </td></tr>

</table></td></tr></table>
</body></html>"""


def build_report(companies_with_rows: list[tuple]) -> str:
    confirmed = likely = clean_signals = 0

    sections = []
    for rank, (company, rows) in enumerate(companies_with_rows, 1):
        for row in rows:
            s = row["situation_status"].lower()
            q = signal_quality(row)
            if "confirmed" in s: confirmed += 1
            elif "likely"  in s: likely    += 1
            if q["quality"] == "clean" and "no signals" not in row.get("detected_signal", "").lower() and row.get("detected_signal", ""):
                clean_signals += 1
        sections.append(build_company_section(company, rows, rank))

    return REPORT_HTML.format(
        date=TODAY,
        cutoff=CUTOFF_STR,
        num_companies=len(companies_with_rows),
        confirmed=confirmed,
        likely=likely,
        clean_signals=clean_signals,
        company_sections="\n".join(sections),
    )


# ── Email ──────────────────────────────────────────────────────────────────────
def send_email(html: str, attachment: Path):
    smtp_user = os.environ["GMAIL_ADDRESS"]
    smtp_pass = os.environ["GMAIL_APP_PASSWORD"]
    to_addr   = os.environ["RECIPIENT_EMAIL"]
    subject   = f"XIMPAX Weekly Intelligence Report — {datetime.utcnow().strftime('%d %b %Y')}"

    msg            = MIMEMultipart("mixed")
    msg["From"]    = smtp_user
    msg["To"]      = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(html, "html", "utf-8"))

    with open(attachment, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{attachment.name}"')
    msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(smtp_user, smtp_pass)
        s.sendmail(smtp_user, to_addr, msg.as_string())
    log.info(f"✅ Email sent to {to_addr}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    companies = load_stage2_input()
    log.info(f"Stage 2: deep scanning {len(companies)} companies | Evidence cutoff: {CUTOFF_STR}")

    companies_with_rows: list[tuple] = []
    raw_all: dict = {}

    for i, company in enumerate(companies):
        try:
            # Call A: live research with Google Search
            research_text = gemini_research(
                build_research_prompt(company), company["company"]
            )
            raw_all[company["company"]] = {"research": research_text}
            time.sleep(5)

            # Call B: format into table
            table_text = gemini_format(research_text, company)
            raw_all[company["company"]]["table"] = table_text

            rows = parse_stage2_table(table_text)
            log.info(f"  → {len(rows)} signal rows for {company['company']}")

            # Log quality breakdown
            clean = sum(1 for r in rows if signal_quality(r)["quality"] == "clean")
            flagged = len(rows) - clean
            log.info(f"     ✅ {clean} clean | ⚠️  {flagged} flagged")

            companies_with_rows.append((company, rows))

        except Exception as e:
            log.warning(f"Failed for {company['company']}: {e}")
            companies_with_rows.append((company, []))

        if i < len(companies) - 1:
            log.info(f"  Sleeping {SLEEP_BETWEEN}s …")
            time.sleep(SLEEP_BETWEEN)

    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"stage2_raw_{ts}.json"
    raw_path.write_text(json.dumps(raw_all, ensure_ascii=False, indent=2))

    html      = build_report(companies_with_rows)
    html_path = OUTPUT_DIR / f"stage2_report_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Stage 2 report → {html_path}")

    try:
        send_email(html, html_path)
    except Exception as e:
        log.error(f"Email failed: {e}")
        raise

    return html_path


if __name__ == "__main__":
    main()
