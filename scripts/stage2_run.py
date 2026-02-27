"""
XIMPAX Intelligence Engine — Stage 2

Architecture per company:
  Call A → Perplexity sonar-pro (live web search, cited sources, real URLs)
  Call B → Gemini flash (formats prose into 4-column signal table, no search)

Why Perplexity for research:
  - Built-in web search with proper citation (Reuters, Bloomberg, FT etc.)
  - Does NOT default to company IR pages or careers sites
  - Returns specific quotes with publication dates
  - More reliable than Gemini+Google Search grounding for structured research

Requires env vars: PERPLEXITY_API_KEY, GEMINI_API_KEY, GMAIL_ADDRESS,
                   GMAIL_APP_PASSWORD, RECIPIENT_EMAIL
"""

import os
import re
import json
import time
import smtplib
import logging
import urllib.request
import urllib.error
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

GEMINI_MODEL      = "gemini-2.0-flash"
PERPLEXITY_MODEL  = "sonar-pro"   # sonar-pro has deeper search; fallback: "sonar"
SLEEP_BETWEEN     = 12            # seconds between companies

CUTOFF_DATE = datetime.utcnow() - timedelta(days=365)
CUTOFF_STR  = CUTOFF_DATE.strftime("%Y-%m-%d")
TODAY       = datetime.utcnow().strftime("%Y-%m-%d")


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_stage2_input() -> list[dict]:
    path = OUTPUT_DIR / "stage2_input.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 input not found at {path}. Run stage1_run.py first.")
    data = json.loads(path.read_text())
    data.sort(key=lambda r: r.get("_score_int", 0), reverse=True)
    log.info(f"Stage 2 input: {len(data)} companies:")
    for r in data:
        log.info(f"  → {r['company']:35s} | {r.get('_score_int','?')}/40")
    return data


# ── CALL A: Perplexity research ────────────────────────────────────────────────
def build_perplexity_prompt(company: dict) -> str:
    instructions  = load_file(CONFIG_DIR / "instructions.txt")
    icp_blueprint = load_file(CONFIG_DIR / "icp_blueprint.txt")

    return f"""You are a senior supply chain intelligence analyst. Today is {TODAY}.

Research this company for supply chain execution signals relevant to a consulting firm
that sells rapid-deployment supply chain execution programs:

Company:  {company['company']}
Country:  {company['hq_country']}
Industry: {company['industry']}
Revenue:  {company['revenue']}
Stage 1 note: {company.get('situations','unknown')} | Score: {company.get('priority_score','unknown')}

══════════════════════════════════════════════════════
SEARCH FOR ALL 4 SITUATIONS — DO NOT SKIP ANY
══════════════════════════════════════════════════════

SITUATION 1 — RESOURCE CONSTRAINTS
Search: "{company['company']} supply chain procurement staffing vacancy capacity gap {TODAY[:4]}"
Search: "{company['company']} CPO supply chain director departure leadership change {TODAY[:4]}"
Search: "{company['company']} ERP S4HANA IBP S&OP transformation delayed resource shortage {TODAY[:4]}"
Look for: leadership churn, hiring freezes, explicit "bandwidth" or "capacity" language from management,
delayed transformation programs, interim/contractor reliance

SITUATION 2 — MARGIN PRESSURE
Search: "{company['company']} EBITDA gross margin decline cost reduction restructuring {TODAY[:4]}"
Search: "{company['company']} profit warning savings program layoffs plant closure {TODAY[:4]}"
Look for: structural margin decline, quantified savings targets, guidance downgrades due to costs,
restructuring programs with headcount or plant closure details

SITUATION 3 — SIGNIFICANT GROWTH
Search: "{company['company']} acquisition merger new plant capacity expansion investment {TODAY[:4]}"
Search: "{company['company']} revenue growth new markets supply chain scaling {TODAY[:4]}"
Look for: M&A requiring SC integration, new manufacturing sites, capacity investments,
revenue growth outpacing operational capability, geographic expansion

SITUATION 4 — SUPPLY CHAIN DISRUPTION
Search: "{company['company']} supply disruption shortage production halt recall quality {TODAY[:4]}"
Look for: production stoppages, product recalls, supplier failures, logistics crises,
missed guidance due to supply issues

══════════════════════════════════════════════════════
STRICT EVIDENCE RULES
══════════════════════════════════════════════════════
REQUIRED: Every piece of evidence must be dated after {CUTOFF_STR}
REQUIRED: Every evidence item must include a real, accessible URL
REQUIRED: Source must be independent media — NOT the company's own website, press room,
  careers page, or investor relations page
PREFERRED sources: Reuters, Bloomberg, Financial Times, Seeking Alpha, trade press
  (Fierce Pharma, Chemical Week, Food Ingredients First, etc.), Swissquote, NZZ,
  GlobeNewswire (3rd party announcements only), Tipranks, Investing.com, MarketWatch

FORBIDDEN sources (do not cite these):
  ✗ {company['company']} company website, press releases, annual reports, newsroom
  ✗ {company['company']} careers/jobs pages
  ✗ LinkedIn job posts
  ✗ PortersFiveForce, Comparably, Craft.co, Macroaxis, SimplyWallSt, MarketBeat
  ✗ Generic industry articles not specifically about {company['company']}
  ✗ Any article dated before {CUTOFF_STR}

SIGNAL WEIGHTS (apply strictly):
  STRONG signal = +2 pts | MEDIUM signal = +1 pt
  STRONG = management explicitly states, quantified data, formal announcement
  MEDIUM = indirect evidence, analyst commentary, supporting data
  Cap per situation = 10 pts max

=== SITUATION SIGNAL DEFINITIONS ===
{instructions}

=== ICP BLUEPRINT ===
{icp_blueprint}

══════════════════════════════════════════════════════
OUTPUT FORMAT — write exactly this structure:
══════════════════════════════════════════════════════

SITUATION 1 — RESOURCE CONSTRAINTS: [CONFIRMED/LIKELY/UNCLEAR/NOT PRESENT] ([X] pts)
Signal: [exact signal name from framework] | Strength: STRONG/MEDIUM | Weight: +2/+1
Quote: "[verbatim quote ≤25 words from the article — not from company's own marketing]"
Date: YYYY-MM-DD | Source: [Publication name] | URL: [full https:// URL]
[repeat per signal, or write: No qualifying evidence found after {CUTOFF_STR} from independent sources]

SITUATION 2 — MARGIN PRESSURE: [classification] ([X] pts)
[same format]

SITUATION 3 — SIGNIFICANT GROWTH: [classification] ([X] pts)
[same format]

SITUATION 4 — SUPPLY CHAIN DISRUPTION: [classification] ([X] pts)
[same format]

TOTAL SCORE: RC:[n] | MP:[n] | SG:[n] | SCD:[n] = [total]/40
"""


# ── CALL B: Gemini formatting ──────────────────────────────────────────────────
def build_format_prompt(company: dict, research_text: str) -> str:
    prompt_template = load_file(PROMPTS_DIR / "prompt_2.txt")
    return f"""You are a data formatting engine. Convert research findings into a markdown table.
Do NOT research further. Do NOT change scores, quotes or URLs. Just reformat.

=== RESEARCH FINDINGS ===
{research_text}

=== TABLE FORMAT REQUIRED ===
{prompt_template}

RULES:
1. Output ONLY the markdown table — no text before or after
2. Every row must start with | and end with |
3. Exactly 4 columns: Situation Status | Detected Signal | Evidence & Quote | Source & URL
4. One row per detected signal
5. Source & URL: use ONLY URLs that appear verbatim in the research above — never invent any
6. Include the publication date in Evidence: "(Published: YYYY-MM-DD)"
7. No evidence for a situation → one row: "No signals detected"
8. Weight labels: STRONG is always +2, MEDIUM is always +1 — never write MEDIUM +2
9. Source label = publication name (Reuters, Bloomberg etc.) not an article headline
"""


# ── Perplexity API call ────────────────────────────────────────────────────────
def call_perplexity(prompt: str, company_name: str) -> str:
    """
    Call Perplexity sonar-pro via its OpenAI-compatible REST API.
    Uses urllib to avoid requiring the openai package.
    """
    api_key = os.environ["PERPLEXITY_API_KEY"]
    log.info(f"  → Call A (Perplexity): researching {company_name} …")

    payload = json.dumps({
        "model": PERPLEXITY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a supply chain market intelligence researcher. "
                    "Always search for the most recent news and cite real URLs. "
                    "Never use company websites, IR pages, or careers pages as sources. "
                    "Every evidence item must have a real accessible URL."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4000,
        "temperature": 0.1,
        "search_recency_filter": "year",   # restrict to last 12 months
        "return_citations": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.perplexity.ai/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Perplexity HTTP {e.code}: {body[:400]}") from e

    text = data["choices"][0]["message"]["content"]
    log.info(f"  → Call A returned {len(text)} chars")

    # Append inline citations if Perplexity returned them separately
    citations = data.get("citations", [])
    if citations:
        text += "\n\n=== CITED SOURCES ===\n"
        for i, url in enumerate(citations, 1):
            text += f"[{i}] {url}\n"

    return text


# ── Gemini formatting call ─────────────────────────────────────────────────────
def call_gemini_format(research_text: str, company: dict) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    log.info(f"  → Call B (Gemini): formatting {company['company']} …")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_format_prompt(company, research_text),
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a data formatting engine. Output ONLY the markdown table. "
                "Start with | header row. Every row starts and ends with |. "
                "No text before or after the table. "
                "Never invent or modify URLs — only use URLs from the input. "
                "MEDIUM signals are always +1. STRONG signals are always +2."
            ),
            temperature=0.0,
            max_output_tokens=8000,
        ),
    )
    text = response.text or ""
    log.info(f"  → Call B returned {len(text)} chars")
    return text


# ── Table parser ───────────────────────────────────────────────────────────────
def parse_stage2_table(raw: str) -> list[dict]:
    raw        = re.sub(r"```[a-z]*", "", raw)
    lines      = [l.strip() for l in raw.splitlines()]
    pipe_lines = [l for l in lines if l.startswith("|")]

    if not pipe_lines:
        log.warning("No pipe-delimited lines in Stage 2 response.")
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
FORBIDDEN_SOURCE_PATTERNS = [
    r"\bcompany\s+website\b", r"\bcompany\s+site\b", r"\bIR\s+page\b",
    r"\binvestor\s+relations\b", r"\bpress\s+release\b", r"\bvendor\s+blog\b",
    r"\bnewsroom\b", r"\bjobs?\s+page\b",
    # Explicit SWOT/aggregator sites
    r"portersfiveforce", r"comparably", r"craft\.co", r"macroaxis",
    r"stockanalysis\.com", r"wisesheets", r"marketbeat", r"simplywall",
    # Z2Data is a generic supply chain risk aggregator — not company-specific
    r"z2data",
]

FORBIDDEN_SOURCE_LABELS = {
    "lonza", "givaudan", "novartis", "roche", "nestle", "abb", "sika",
    "sonova", "straumann", "georg fischer", "lindt", "emmi", "dätwyler",
    "tecan", "bossard", "orior", "huber", "bucher", "bobst", "bachem",
    "siegfried", "ypsomed", "tecan", "clariant", "ems-chemie", "komax",
    "bystronic", "endress", "sensirion", "skan", "coltene",
}

COMPANY_PAGE_SUFFIXES = {
    "careers", "career", "jobs", "newsroom", "press", "investor",
    "investors", "ir", "annual report", "sustainability report",
}


def classify_source(source_str: str) -> str:
    s = source_str.lower().strip()

    # 1. Explicit forbidden patterns
    for pat in FORBIDDEN_SOURCE_PATTERNS:
        if re.search(pat, s, re.IGNORECASE):
            return "forbidden"

    # 2. Bare company name or company name + page suffix
    for label in FORBIDDEN_SOURCE_LABELS:
        escaped = re.escape(label)
        if re.match(rf"^{escaped}(\s*[\(\[]|$)", s):
            return "forbidden"
        for suffix in COMPANY_PAGE_SUFFIXES:
            if re.match(rf"^{escaped}\s+{re.escape(suffix)}", s):
                return "forbidden"

    # 3. Source is clearly a URL-as-label starting with the company domain
    #    e.g. "lonza.com/news/..." — extract domain and check
    domain_m = re.match(r"https?://([^/]+)", s)
    if domain_m:
        domain = domain_m.group(1).lower().replace("www.", "")
        for label in FORBIDDEN_SOURCE_LABELS:
            if domain.startswith(label.replace("+", "").replace(" ", "")):
                return "forbidden"

    # NOTE: We intentionally do NOT flag long source labels as "headline detection"
    # because Perplexity often returns article titles as source names for GlobeNewswire
    # press releases etc. The quality of Perplexity citations is much higher so
    # we rely on the URL itself being a real external domain.

    return "clean"


def parse_evidence_date(evidence: str) -> datetime | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", evidence)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass
    return None


def signal_quality(row: dict) -> dict:
    ev_date   = parse_evidence_date(row.get("evidence", ""))
    src_class = classify_source(row.get("source_url", ""))
    stale     = ev_date is not None and ev_date < CUTOFF_DATE
    forbidden = src_class == "forbidden"

    if stale and forbidden:  quality = "stale+forbidden"
    elif stale:              quality = "stale"
    elif forbidden:          quality = "forbidden"
    else:                    quality = "clean"

    return {"stale": stale, "forbidden": forbidden, "quality": quality}


# ── Score recalculation (clean signals only) ───────────────────────────────────
def recalculate_score(rows: list[dict]) -> int:
    sit_scores: dict[str, int] = {
        "resource constraints": 0, "margin pressure": 0,
        "significant growth": 0, "supply chain disruption": 0,
    }
    current = ""
    for row in rows:
        if row["situation_status"].strip():
            current = row["situation_status"].lower()
        sig = row.get("detected_signal", "").lower()
        if "no signals" in sig or not sig:
            continue
        if signal_quality(row)["quality"] != "clean":
            continue
        w = 2 if "strong" in sig else (1 if "medium" in sig else 0)
        for k in sit_scores:
            if k in current:
                sit_scores[k] = min(10, sit_scores[k] + w)
                break
    return sum(sit_scores.values())


# ── HTML builder ───────────────────────────────────────────────────────────────
SITUATION_STYLES = {
    "confirmed":   ("#86efac", "#14532d", "✅ CONFIRMED"),
    "likely":      ("#93c5fd", "#1e3a5f", "🔵 LIKELY"),
    "unclear":     ("#fcd34d", "#422006", "⚠️ UNCLEAR"),
    "not present": ("#9ca3af", "#1f2937", "⬜ NOT PRESENT"),
}

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


def situation_style(s: str):
    t = s.lower()
    for k, v in SITUATION_STYLES.items():
        if k in t:
            return v
    return ("#d1d5db", "#1f2937", s[:40])


def make_source_link(src: str, quality: str) -> str:
    urls  = re.findall(r"https?://[^\s<>\"'\]]+", src)
    label = re.sub(r"https?://[^\s<>\"'\]]+", "", src).strip(" —-–[]")
    badge_html, _ = QUALITY_BADGES.get(quality, ("", ""))

    if urls:
        link = f'<a href="{urls[0]}" target="_blank" style="color:#60a5fa">{label or "🔗 Source"}</a>'
    elif src.strip() and src.strip() not in ("-", "—"):
        link = f'<span style="color:#6b7280;font-size:10px">{src[:80]}</span>'
    else:
        link = '<span style="color:#6b7280;font-size:10px">No URL</span>'
    return link + badge_html


def build_company_section(company: dict, rows: list[dict], rank: int) -> str:
    verified_score = recalculate_score(rows)
    raw_score      = company.get("_score_int", 0)
    bar_pct        = min(100, int(verified_score / 40 * 100))
    dom_color      = "#16a34a" if any("confirmed" in r["situation_status"].lower() for r in rows) else "#2563eb"

    total_sig = sum(1 for r in rows if r.get("detected_signal","") and "no signals" not in r["detected_signal"].lower())
    clean_sig = sum(1 for r in rows if signal_quality(r)["quality"] == "clean"
                    and r.get("detected_signal","") and "no signals" not in r["detected_signal"].lower())
    flag_sig  = total_sig - clean_sig

    quality_line = (
        f'<span style="font-size:11px;color:#6b7280;margin-top:4px;display:block">'
        f'✅ {clean_sig} clean signals'
        + (f' &nbsp;|&nbsp; <span style="color:#fca5a5">⚠️ {flag_sig} flagged</span>' if flag_sig else "")
        + '</span>'
    )

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
            q         = signal_quality(r)
            _, row_sty = QUALITY_BADGES.get(q["quality"], ("", ""))
            sit_td = (
                f'<td style="background:{bg};color:{tc};font-weight:700;font-size:11px;'
                f'white-space:nowrap;padding:10px 8px;border-right:1px solid #374151">'
                f'{label}<br><small style="font-size:9px;font-weight:400;opacity:.8">{sit}</small></td>'
            ) if first else '<td style="background:#111827;border-right:1px solid #374151"></td>'
            first = False
            sig_rows_html.append(f"""
            <tr style="border-bottom:1px solid #374151;{row_sty}">
              {sit_td}
              <td style="padding:10px 8px;font-size:12px;color:#d1d5db;border-right:1px solid #374151">{r['detected_signal']}</td>
              <td style="padding:10px 8px;font-size:11px;font-style:italic;color:#9ca3af;border-right:1px solid #374151">{r['evidence']}</td>
              <td style="padding:10px 8px;font-size:11px">{make_source_link(r['source_url'], q['quality'])}</td>
            </tr>""")

    no_sig = ('<tr><td colspan="4" style="padding:14px;color:#6b7280;text-align:center;'
              'font-style:italic">No signals found within last 12 months from independent sources</td></tr>')

    return f"""
    <div style="background:#1f2937;border-radius:10px;margin-bottom:24px;overflow:hidden;border:1px solid #374151">
      <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-left:4px solid {dom_color};
                  padding:16px 20px;display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">#{rank} Priority Lead</div>
          <h2 style="font-size:18px;margin:0;color:#f9fafb;font-weight:700">{company['company']}</h2>
          <p style="font-size:12px;color:#9ca3af;margin:5px 0 2px">{company['industry']} &nbsp;·&nbsp; {company['hq_country']} &nbsp;·&nbsp; {company['revenue']}</p>
          {quality_line}
        </div>
        <div style="text-align:right;min-width:140px">
          <div style="font-size:30px;font-weight:800;color:#60a5fa">{verified_score}<small style="font-size:14px;color:#6b7280">/40</small></div>
          <div style="background:rgba(255,255,255,.1);border-radius:4px;height:6px;width:110px;margin:6px 0 2px;margin-left:auto">
            <div style="background:linear-gradient(90deg,#2563eb,#7c3aed);height:6px;border-radius:4px;width:{bar_pct}%"></div>
          </div>
          <small style="font-size:10px;color:#6b7280">Verified (clean signals only)</small>
          {f'<small style="font-size:10px;color:#6b7280;display:block">S1 estimate: {raw_score}/40</small>' if raw_score != verified_score else ""}
        </div>
      </div>
      <div style="background:#1a2535;padding:6px 16px;border-bottom:1px solid #374151;font-size:10px;color:#6b7280">
        ⏰ STALE = older than {CUTOFF_STR} &nbsp;|&nbsp; 🚫 EXCL.SOURCE = company website / IR page
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
    <p style="margin:8px 0 0;color:#9ca3af;font-size:13px">Stage 2 · {date} · {num_companies} Companies · Powered by Perplexity sonar-pro</p>
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
    XIMPAX Intelligence Engine · Research: Perplexity sonar-pro · Formatting: Gemini 2.0 Flash<br>
    Evidence window: {cutoff} → {date}
  </td></tr>
</table></td></tr></table>
</body></html>"""


def build_report(companies_with_rows: list[tuple]) -> str:
    confirmed = likely = clean_signals = 0
    sections  = []

    for rank, (company, rows) in enumerate(companies_with_rows, 1):
        for row in rows:
            s = row["situation_status"].lower()
            q = signal_quality(row)
            if "confirmed" in s: confirmed += 1
            elif "likely"  in s: likely    += 1
            if (q["quality"] == "clean" and row.get("detected_signal","")
                    and "no signals" not in row["detected_signal"].lower()):
                clean_signals += 1
        sections.append(build_company_section(company, rows, rank))

    return REPORT_HTML.format(
        date=TODAY, cutoff=CUTOFF_STR,
        num_companies=len(companies_with_rows),
        confirmed=confirmed, likely=likely, clean_signals=clean_signals,
        company_sections="\n".join(sections),
    )


# ── Email ──────────────────────────────────────────────────────────────────────
def send_email(html: str, attachment: Path):
    smtp_user = os.environ["GMAIL_ADDRESS"]
    smtp_pass = os.environ["GMAIL_APP_PASSWORD"]
    to_addr   = os.environ["RECIPIENT_EMAIL"]
    subject   = f"XIMPAX Weekly Intelligence Report — {datetime.utcnow().strftime('%d %b %Y')}"

    msg = MIMEMultipart("mixed")
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
    log.info(f"Stage 2: researching {len(companies)} companies via Perplexity sonar-pro")
    log.info(f"Evidence window: {CUTOFF_STR} → {TODAY}")

    companies_with_rows: list[tuple] = []
    raw_all: dict = {}

    for i, company in enumerate(companies):
        log.info(f"[{i+1}/{len(companies)}] {company['company']}")
        try:
            # Call A: Perplexity live research
            research_text = call_perplexity(
                build_perplexity_prompt(company), company["company"]
            )
            raw_all[company["company"]] = {"research": research_text}
            time.sleep(4)

            # Call B: Gemini format into table
            table_text = call_gemini_format(research_text, company)
            raw_all[company["company"]]["table"] = table_text

            rows = parse_stage2_table(table_text)
            clean = sum(1 for r in rows if signal_quality(r)["quality"] == "clean")
            flagged = len(rows) - clean
            log.info(f"  → {len(rows)} rows | ✅ {clean} clean | ⚠️  {flagged} flagged")

            companies_with_rows.append((company, rows))

        except Exception as e:
            log.warning(f"  Failed: {e}")
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
