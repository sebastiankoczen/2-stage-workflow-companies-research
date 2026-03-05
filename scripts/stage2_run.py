"""
XIMPAX Intelligence Engine — Stage 2 (v5 — Gmail + chain substitution)

CHANGES FROM v4:
  - NEW: If a reserve substitute itself scores 0 (all-stale evidence), pull
    one more reserve rather than including a 0-score company in the report.
    This prevents the u-blox situation (reserve pulled for Vencorex, all
    signals stale, appeared as 0/40 in final report).
    Only applies to absolute zero (verified=0). No further chain after this.

CHANGES FROM v3 (v4):
  - NEW: Auto-substitution. Stage 1 now saves 20 companies (primary 10 +
    reserve 10). After deep-scanning each primary, Stage 2 checks whether
    it's a "false signal" using THREE conditions:

      1. ABSOLUTE FLOOR   — verified score < SUBSTITUTION_VERIFIED_MIN (6)
      2. RATIO TRIGGER    — verified/s1 < 30% AND verified < 10
      3. EVIDENCE THIN    — Call A returned < CALL_A_MIN_CHARS (2500)
                            Gemini finding almost nothing = no real evidence exists

    If any condition fires (and S1 score was >= 12), the company is replaced
    by the next reserve in the queue. The report shows substituted companies
    in an audit trail section.

  - FIX: Subject line changed to stable prefix format for Power Automate
    compatibility:  "XIMPAX Weekly Intelligence Report | YYYY-MM-DD"
"""

import os
import re
import json
import time
import logging
import smtplib
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
SLEEP_BETWEEN = 12  # seconds between companies

CUTOFF_DATE = datetime.utcnow() - timedelta(days=365)
CUTOFF_STR  = CUTOFF_DATE.strftime("%Y-%m-%d")
TODAY       = datetime.utcnow().strftime("%Y-%m-%d")

# ── Substitution thresholds ────────────────────────────────────────────────────
# A primary is replaced if ANY of these fires (and S1 score >= 12):
SUBSTITUTION_VERIFIED_MIN = 6     # 1. absolute floor: verified score below this
SUBSTITUTION_RATIO        = 0.30  # 2. ratio: verified/s1 < 30% AND verified < 10
CALL_A_MIN_CHARS          = 2500  # 3. evidence thin: Call A response below this
                                  #    = Gemini found almost nothing in search

# ── Source quality config ─────────────────────────────────────────────────────
FORBIDDEN_SOURCE_PATTERNS = [
    r"\bcompany\s+website\b", r"\bcompany\s+site\b", r"\bIR\s+page\b",
    r"\binvestor\s+relations\b", r"\bpress\s+release\b",
    r"\bvendor\s+blog\b", r"\bswot\b",
    r"\bcareers?\b", r"\bjobs?\s+page\b", r"\bnewsroom\b",
    r"portersfiveforce", r"comparably", r"craft\.co", r"macroaxis",
    r"stockanalysis\.com", r"wisesheets", r"marketbeat", r"simplywall",
]

FORBIDDEN_SOURCE_LABELS = {
    "lonza", "givaudan", "novartis", "roche", "nestle", "abb", "sika",
    "sonova", "straumann", "georg fischer", "lindt", "emmi", "datwyler",
    "tecan", "bossard", "orior", "huber", "bucher", "bobst",
    "campari", "biogen", "moderna", "vetoquinol", "sun pharma",
    "huntsman", "integra", "smith nephew", "dexcom", "ciba",
    "celanese", "chemours", "brenntag", "swarovski", "idorsia",
}

COMPANY_PAGE_SUFFIXES = {
    "careers", "career", "jobs", "newsroom", "press", "news",
    "investor", "investors", "ir", "annual report", "sustainability report",
}

GENERIC_ARTICLE_PATTERNS = [
    r"procurement teams will be tested",
    r"supply chain professionals will find",
    r"global supply chain workforce shortage",
    r"supply chain managers will confront",
    r"geopolitical tensions continue to disrupt global supply chains",
    r"supply chain management has emerged as the dominant strategic priority",
    r"port disruptions from infrastructure",
    r"biggest obstacles in 20\d\d",
    r"everstream forecasts",
    r"companies in this sector are so busy",
    r"seven plus years of progressive leadership",
    r"logistics is constantly evolving",
    r"trade professionals.*nearly double",
    r"rising costs and shifting trade dynamics",
    r"in 2026, supply chain managers",
    r"geopolitics has re-emerged as one of the biggest disruptors",
]


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_stage2_input() -> tuple[list[dict], list[dict]]:
    """Returns (primary_companies, reserve_companies)."""
    path = OUTPUT_DIR / "stage2_input.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 input not found at {path}.")
    data = json.loads(path.read_text())

    primary = [r for r in data if r.get("_priority", "primary") == "primary"]
    reserve = [r for r in data if r.get("_priority") == "reserve"]

    # Fallback: Stage 1 not yet updated — treat all as primary, no reserve
    if not reserve:
        log.warning(
            "No reserve companies in stage2_input.json. "
            "Update stage1_run.py to save primary+reserve (see stage1_patch.py). "
            "Running without substitution this time."
        )
        primary = data
        reserve = []

    primary.sort(key=lambda r: r.get("_priority_rank", 99))
    reserve.sort(key=lambda r: r.get("_priority_rank", 99))

    log.info(f"Loaded {len(primary)} primary + {len(reserve)} reserve companies")
    log.info("PRIMARY:")
    for r in primary:
        log.info(f"  [{r.get('_priority_rank','?'):>2}] {r['company']:35s} | {r.get('_score_int','?')}/40")
    if reserve:
        log.info("RESERVE (substitution pool):")
        for r in reserve:
            log.info(f"  [{r.get('_priority_rank','?'):>2}] {r['company']:35s} | {r.get('_score_int','?')}/40")
    return primary, reserve


# ── Substitution decision ──────────────────────────────────────────────────────
def is_false_signal(company: dict, rows: list[dict],
                    verified_score: int, call_a_chars: int) -> tuple[bool, str]:
    """
    Returns (should_substitute: bool, reason: str).

    Three conditions — any one fires substitution (if S1 >= 12):
      1. verified_score < SUBSTITUTION_VERIFIED_MIN  (absolute floor)
      2. verified/s1 < SUBSTITUTION_RATIO AND verified < 10  (ratio drop)
      3. call_a_chars < CALL_A_MIN_CHARS  (Gemini found almost nothing)
    """
    s1 = company.get("_score_int", 0)

    # Don't substitute low-S1 companies — low score → low evidence is expected
    if s1 < 12:
        return False, ""

    if verified_score < SUBSTITUTION_VERIFIED_MIN:
        reason = f"absolute floor (verified={verified_score} < {SUBSTITUTION_VERIFIED_MIN})"
        log.info(f"  --> FALSE SIGNAL [{reason}]: {company['company']}")
        return True, reason

    if s1 > 0 and (verified_score / s1) < SUBSTITUTION_RATIO and verified_score < 10:
        ratio_pct = round(verified_score / s1 * 100)
        reason = f"ratio trigger (verified={verified_score}/s1={s1} = {ratio_pct}% < {int(SUBSTITUTION_RATIO*100)}%)"
        log.info(f"  --> FALSE SIGNAL [{reason}]: {company['company']}")
        return True, reason

    if call_a_chars < CALL_A_MIN_CHARS:
        reason = f"evidence thin (Call A={call_a_chars} chars < {CALL_A_MIN_CHARS} threshold)"
        log.info(f"  --> FALSE SIGNAL [{reason}]: {company['company']}")
        return True, reason

    return False, ""


# ── Research + format prompts ──────────────────────────────────────────────────
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

======================================================
MANDATORY DATE RULE - ZERO EXCEPTIONS
======================================================
The cutoff date is {CUTOFF_STR}. Every single piece of evidence MUST be
dated AFTER {CUTOFF_STR}. If you find an article and its publication date
is before {CUTOFF_STR}, you MUST discard it and search for a newer source.
Do NOT use any source older than 12 months. This is non-negotiable.

======================================================
MANDATORY SOURCE RULES - ZERO EXCEPTIONS
======================================================
ALLOWED sources (use these):
  Tier 1: Reuters, Bloomberg, Financial Times, regulatory filings,
          earnings call transcripts (Seeking Alpha, Motley Fool, etc.)
  Tier 2: Trade publications (Pharma Tech, Chemical Week, etc.),
          Swissquote, NZZ, Neue Zurcher Zeitung, Swiss business press
  Tier 3: Reputable business news (Forbes, WSJ, etc.)

FORBIDDEN - do NOT use any of these:
  x {company['company']} company website, press releases, newsroom, or careers page
  x Any company's own investor relations page
  x LinkedIn job posts (only acceptable for Organizational signals,
    and ONLY if the posting itself is dated after {CUTOFF_STR})
  x Generic industry blogs or vendor content
  x ZipRecruiter, Indeed, or other job boards for non-job-specific signals
  x SWOT or AI analysis aggregator sites: PortersFiveForce.com, Comparably,
    Craft.co, Macroaxis, StockAnalysis, WiseSheets, MarketBeat, SimplyWallSt
  x Generic industry trend articles that do NOT explicitly name
    {company['company']} as the subject of the reported fact.

STRICT COUNTING RULES:
  x Same article/URL may only count for ONE signal
  x Maximum 2 signals from any single publication per situation
  x Do NOT count a signal without a real confirmed URL - write "no evidence found"

======================================================
SEARCH QUERIES - RUN ALL OF THESE
======================================================
SITUATION 1 - RESOURCE CONSTRAINTS:
  Search: "{company['company']} supply chain procurement staffing shortage capacity gap {TODAY[:4]}"
  Search: "{company['company']} CPO supply chain director departure vacancy hire {TODAY[:4]}"
  Search: "{company['company']} ERP SAP S4 IBP transformation delayed resource {TODAY[:4]}"

SITUATION 2 - MARGIN PRESSURE:
  Search: "{company['company']} EBITDA gross margin decline cost reduction restructuring {TODAY[:4]}"
  Search: "{company['company']} profit warning savings program procurement efficiency {TODAY[:4]}"
  Search: "{company['company']} layoffs plant closure footprint optimization {TODAY[:4]}"

SITUATION 3 - SIGNIFICANT GROWTH:
  Search: "{company['company']} acquisition merger plant expansion capacity investment {TODAY[:4]}"
  Search: "{company['company']} revenue growth supply chain scaling backlog {TODAY[:4]}"

SITUATION 4 - SUPPLY CHAIN DISRUPTION:
  Search: "{company['company']} supply disruption shortage production halt recall {TODAY[:4]}"
  Search: "{company['company']} supplier failure logistics disruption quality crisis {TODAY[:4]}"

======================================================
SCORING RULES
======================================================
STRONG signal = +2 pts | MEDIUM signal = +1 pt | Cap per situation = 10 pts
CONFIRMED >= 10 pts | LIKELY 6-9 pts | UNCLEAR 3-5 pts | NOT PRESENT 0-2 pts

=== SITUATION SIGNAL DEFINITIONS ===
{instructions}

=== ICP BLUEPRINT ===
{icp_blueprint}

======================================================
OUTPUT FORMAT
======================================================
Write one block per situation:

SITUATION 1 - RESOURCE CONSTRAINTS: [CONFIRMED/LIKELY/UNCLEAR/NOT PRESENT] ([X] pts)
Signal: [exact signal name] | Strength: STRONG/MEDIUM | Weight: +2/+1
Quote: "[verbatim quote, max 25 words, must name {company['company']} explicitly]"
Date: YYYY-MM-DD (must be after {CUTOFF_STR}) | Source: [Publication name] | URL: [full https:// URL]
[repeat for each signal, or write "No evidence found within the last 12 months from allowed sources"]

SITUATION 2 - MARGIN PRESSURE: [classification] ([X] pts)
[same format]

SITUATION 3 - SIGNIFICANT GROWTH: [classification] ([X] pts)
[same format]

SITUATION 4 - SUPPLY CHAIN DISRUPTION: [classification] ([X] pts)
[same format]

TOTAL SCORE: RC:[n] | MP:[n] | SG:[n] | SCD:[n] = [total]/40
"""


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
1. Output ONLY the markdown table - no text before or after
2. Every row must start with | and end with |
3. Exactly 4 columns: Situation Status | Detected Signal | Evidence & Quote | Source & URL
4. One row per signal detected
5. In the Source & URL column: use ONLY URLs that appear verbatim in the research above
6. Include the publication date in the Evidence column: "(Published: YYYY-MM-DD)"
7. If research shows "No evidence found" for a situation, write one row with "No signals detected"
8. Signal weight labels MUST be exactly one of:
   "STRONG +2"  (for strong signals worth 2 points)
   "MEDIUM +1"  (for medium signals worth 1 point)
   NEVER write "MEDIUM +2". Medium is always +1.
9. Source label must be the publication name, NOT the article headline.
"""


# ── Gemini calls ───────────────────────────────────────────────────────────────
def gemini_research(prompt: str, company_name: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    log.info(f"  -> Call A: researching {company_name} with Google Search ...")
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
    log.info(f"  -> Call A: {len(text)} chars returned")
    if len(text) < CALL_A_MIN_CHARS:
        log.warning(
            f"  -> ⚠️  THIN EVIDENCE: {len(text)} chars (threshold={CALL_A_MIN_CHARS}) "
            f"— substitution may trigger"
        )
    return text


def gemini_format(research_text: str, company: dict) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    log.info(f"  -> Call B: formatting {company['company']} into table ...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_format_prompt(company, research_text),
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a data formatting engine. Output ONLY the markdown table. "
                "Start with the | character of the header row. "
                "Every row must start and end with |. "
                "No text before or after the table. "
                "Never invent or modify URLs. "
                "Always include the publication date in the Evidence column."
            ),
            temperature=0.0,
            max_output_tokens=8000,
        ),
    )
    text = response.text or ""
    log.info(f"  -> Call B: {len(text)} chars returned")
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
        while len(cells) < 4:
            cells.append("")
        rows.append({
            "situation_status": cells[0],
            "detected_signal":  cells[1],
            "evidence":         cells[2],
            "source_url":       cells[3],
        })
    return rows


# ── Signal quality ─────────────────────────────────────────────────────────────
def parse_evidence_date(evidence: str) -> datetime | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", evidence)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass
    return None


def classify_source(source_str: str) -> str:
    s = source_str.lower().strip()
    for pat in FORBIDDEN_SOURCE_PATTERNS:
        if re.search(pat, s, re.IGNORECASE):
            return "forbidden"
    for label in FORBIDDEN_SOURCE_LABELS:
        escaped = re.escape(label)
        if re.match(rf"^{escaped}(\s*[\(\[]|$|\.com|\.ch|\.de|\.fr|\.co\.uk)", s):
            return "forbidden"
        for suffix in COMPANY_PAGE_SUFFIXES:
            if re.match(rf"^{escaped}\s+{re.escape(suffix)}\b", s):
                return "forbidden"
    trailing_m = re.search(r"[-–|]\s*(.+)$", s)
    if trailing_m:
        trailing = trailing_m.group(1).strip()
        for label in FORBIDDEN_SOURCE_LABELS:
            if label in trailing:
                return "forbidden"
    if re.search(r"\bnews\s+release\b|\bpress\s+release\b|\bofficial\s+statement\b", s):
        return "forbidden"
    PR_VERBS = r"delivers?|reports?|signs?|announces?|confirms?|launches?|achieves?|posts?"
    has_trailing_publisher = bool(re.search(r"\s[-–|]\s*\w", s))
    if not has_trailing_publisher:
        for label in FORBIDDEN_SOURCE_LABELS:
            escaped = re.escape(label)
            if re.match(rf"^{escaped}\s+(?:{PR_VERBS})\b", s):
                return "forbidden"
    return "clean"


def signal_quality(row: dict, company_name: str = "") -> dict:
    ev_date   = parse_evidence_date(row.get("evidence", ""))
    src_class = classify_source(row.get("source_url", ""))
    stale     = ev_date is not None and ev_date < CUTOFF_DATE
    forbidden = src_class == "forbidden"
    combined  = (row.get("evidence", "") + " " + row.get("detected_signal", "")).lower()
    generic   = any(re.search(pat, combined) for pat in GENERIC_ARTICLE_PATTERNS)
    if stale and forbidden:  quality = "stale+forbidden"
    elif stale:              quality = "stale"
    elif forbidden:          quality = "forbidden"
    elif generic:            quality = "generic"
    else:                    quality = "clean"
    return {"stale": stale, "forbidden": forbidden, "generic": generic, "quality": quality}


def recalculate_score(rows: list[dict], company_name: str = "") -> int:
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
        q = signal_quality(row, company_name)
        if q["quality"] in ("stale", "forbidden", "stale+forbidden"):
            continue
        w = 2 if "+2" in sig else (1 if "+1" in sig else 0)
        for k in sit_scores:
            if k in current:
                sit_scores[k] = min(10, sit_scores[k] + w)
                break
    return sum(sit_scores.values())


# ── Deep scan one company ──────────────────────────────────────────────────────
def deep_scan_company(company: dict,
                      sleep_after: bool = True) -> tuple[list[dict], str, str, int]:
    """Run Call A + Call B. Returns (rows, research_text, table_text, call_a_chars)."""
    research_text = gemini_research(build_research_prompt(company), company["company"])
    call_a_chars  = len(research_text)
    time.sleep(5)
    table_text = gemini_format(research_text, company)
    rows       = parse_stage2_table(table_text)

    cname   = company["company"]
    clean   = sum(1 for r in rows if signal_quality(r, cname)["quality"] == "clean")
    flagged = len(rows) - clean
    log.info(f"  -> {len(rows)} signal rows | {clean} clean | {flagged} flagged")

    if sleep_after:
        log.info(f"  Sleeping {SLEEP_BETWEEN}s ...")
        time.sleep(SLEEP_BETWEEN)

    return rows, research_text, table_text, call_a_chars


# ── HTML helpers ───────────────────────────────────────────────────────────────
SITUATION_STYLES = {
    "confirmed":   ("#86efac", "#14532d", "CONFIRMED"),
    "likely":      ("#93c5fd", "#1e3a5f", "LIKELY"),
    "unclear":     ("#fcd34d", "#422006", "UNCLEAR"),
    "not present": ("#9ca3af", "#1f2937", "NOT PRESENT"),
}

def situation_style(s: str):
    t = s.lower()
    for k, v in SITUATION_STYLES.items():
        if k in t: return v
    return ("#d1d5db", "#1f2937", s[:40])

QUALITY_BADGES = {
    "clean":          ("", ""),
    "stale":          (
        ' <span style="background:#7f1d1d;color:#fca5a5;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:700">STALE</span>',
        "border-left:3px solid #ef4444;",
    ),
    "forbidden":      (
        ' <span style="background:#78350f;color:#fcd34d;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:700">EXCL.SOURCE</span>',
        "border-left:3px solid #f59e0b;",
    ),
    "stale+forbidden":(
        ' <span style="background:#7f1d1d;color:#fca5a5;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:700">STALE</span>'
        ' <span style="background:#78350f;color:#fcd34d;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:700">EXCL.SOURCE</span>',
        "border-left:3px solid #ef4444;opacity:0.7;",
    ),
    "generic":        (
        ' <span style="background:#1e293b;color:#64748b;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:700">GENERIC</span>',
        "",
    ),
}

SITUATION_ORDER = [
    "resource constraints", "supply chain disruption",
    "margin pressure",      "significant growth",
]

def make_source_link(src: str, quality: str) -> str:
    urls  = re.findall(r"https?://[^\s<>\"'\]]+", src)
    label = re.sub(r"https?://[^\s<>\"'\]]+", "", src).strip(" —-–[]")
    badge, _ = QUALITY_BADGES.get(quality, ("", ""))
    if urls:
        url = urls[0]
        is_grounded = "vertexaisearch.cloud.google.com" in url or "googleapis.com" in url
        url_badge = (
            ' <span style="background:#1e3a5f;color:#93c5fd;font-size:8px;padding:1px 4px;border-radius:3px;font-weight:700">verified</span>'
            if is_grounded else
            ' <span style="background:#374151;color:#9ca3af;font-size:8px;padding:1px 4px;border-radius:3px;font-weight:700">unverified</span>'
        )
        link = (
            f'<a href="{url}" target="_blank" style="color:#60a5fa;text-decoration:none">'
            f'{label or "Source"}</a>{url_badge}'
        )
    elif src.strip() and src.strip() not in ("-", "—", "No source", ""):
        link = f'<span style="color:#6b7280;font-size:10px">{src[:80]}</span>'
    else:
        link = '<span style="color:#4b5563;font-size:10px;font-style:italic">No source</span>'
    return link + badge


def build_company_section(company: dict, rows: list[dict], rank: int,
                          is_substitute: bool = False,
                          replaced_company: dict | None = None,
                          sub_reason: str = "") -> str:
    cname          = company.get("company", "")
    verified_score = recalculate_score(rows, cname)
    raw_score      = company.get("_score_int", 0)
    bar_pct        = min(100, int(verified_score / 40 * 100))
    dom_color      = "#16a34a" if any("confirmed" in r["situation_status"].lower() for r in rows) else "#2563eb"

    total_sig = sum(1 for r in rows if "no signals" not in r.get("detected_signal","").lower() and r.get("detected_signal",""))
    clean_sig = sum(1 for r in rows if signal_quality(r, cname)["quality"] == "clean" and "no signals" not in r.get("detected_signal","").lower())
    flag_sig  = total_sig - clean_sig

    sub_banner = ""
    if is_substitute and replaced_company:
        orig_name = replaced_company.get("company", "")
        orig_s1   = replaced_company.get("_score_int", "?")
        sub_banner = (
            f'<div style="background:#1a2a1a;border-left:3px solid #f59e0b;padding:7px 14px;'
            f'font-size:10px;color:#fbbf24;margin-bottom:0;border-radius:8px 8px 0 0">'
            f'SUBSTITUTED — replaced {orig_name} (S1={orig_s1}/40 · {sub_reason})'
            f'</div>'
        )

    quality_summary = (
        f'<span style="font-size:11px;color:#6b7280;margin-top:4px;display:block">'
        f'{clean_sig} clean signals'
        + (f' &nbsp;|&nbsp; <span style="color:#fca5a5">{flag_sig} flagged</span>' if flag_sig else "")
        + '</span>'
    )

    groups: dict[str, list] = {}
    cur = ""
    for row in rows:
        if row["situation_status"].strip(): cur = row["situation_status"]
        groups.setdefault(cur, []).append(row)
    orphans = groups.pop("", [])
    if orphans and groups:
        first_key = next(iter(groups))
        groups[first_key] = orphans + groups[first_key]

    def sit_sort_key(sit_label: str) -> int:
        s = sit_label.lower()
        for i, canonical in enumerate(SITUATION_ORDER):
            if canonical in s: return i
        return 99
    ordered_groups = sorted(groups.items(), key=lambda kv: sit_sort_key(kv[0]))

    sig_rows_html = []
    for sit, sig_rows in ordered_groups:
        tc, bg, label = situation_style(sit)
        real_rows    = [r for r in sig_rows if r.get("detected_signal","").strip() and "no signals" not in r.get("detected_signal","").lower()]
        display_rows = real_rows if real_rows else sig_rows[:1]
        rowspan      = len(display_rows)
        for i, r in enumerate(display_rows):
            q = signal_quality(r, cname)
            _, row_sty = QUALITY_BADGES.get(q["quality"], ("",""))
            sit_td = (
                f'<td rowspan="{rowspan}" style="background:{bg};color:{tc};font-weight:700;'
                f'font-size:11px;white-space:nowrap;padding:12px 10px;'
                f'border-right:1px solid #374151;vertical-align:middle;text-align:center;'
                f'border-bottom:2px solid #1e293b;min-width:140px">'
                f'{label}<br><small style="font-size:9px;font-weight:400;opacity:.75;display:block;margin-top:3px">'
                f'{sit.split(":")[0].strip()}</small></td>'
            ) if i == 0 else ""
            sig_rows_html.append(f"""
            <tr style="border-bottom:1px solid #2d3748;{row_sty}">
              {sit_td}
              <td style="padding:10px 8px;font-size:12px;color:#d1d5db;border-right:1px solid #374151;vertical-align:top">{r['detected_signal']}</td>
              <td style="padding:10px 8px;font-size:11px;font-style:italic;color:#9ca3af;border-right:1px solid #374151;vertical-align:top">{r['evidence']}</td>
              <td style="padding:10px 8px;font-size:11px;vertical-align:top">{make_source_link(r['source_url'], q['quality'])}</td>
            </tr>""")

    no_sig = '<tr><td colspan="4" style="padding:14px;color:#6b7280;text-align:center;font-style:italic">No signals detected within allowed sources and date range</td></tr>'

    border_color = "#f59e0b" if is_substitute else "#374151"
    radius = "0 0 10px 10px" if sub_banner else "10px"

    return f"""
    {sub_banner}
    <div style="background:#1f2937;border-radius:{radius};margin-bottom:24px;overflow:hidden;border:1px solid {border_color}">
      <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-left:4px solid {dom_color};
                  padding:16px 20px;display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">#{rank} Priority Lead</div>
          <h2 style="font-size:18px;margin:0;color:#f9fafb;font-weight:700">{cname}</h2>
          <p style="font-size:12px;color:#9ca3af;margin:5px 0 2px">{company['industry']} &nbsp;·&nbsp; {company['hq_country']} &nbsp;·&nbsp; {company['revenue']}</p>
          {quality_summary}
        </div>
        <div style="text-align:right;min-width:160px">
          <div style="font-size:30px;font-weight:800;color:#60a5fa">{verified_score}<small style="font-size:14px;color:#6b7280">/40</small></div>
          <div style="background:rgba(255,255,255,.1);border-radius:4px;height:6px;width:110px;margin:6px 0 2px;margin-left:auto">
            <div style="background:linear-gradient(90deg,#2563eb,#7c3aed);height:6px;border-radius:4px;width:{bar_pct}%"></div>
          </div>
          <small style="font-size:10px;color:#6b7280">Stage 2 verified score</small>
          <small style="font-size:10px;color:#6b7280;display:block">Stage 1 score: {raw_score}/40</small>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;font-family:Arial,sans-serif">
        <thead>
          <tr style="background:#0f172a;border-bottom:2px solid #374151">
            <th style="padding:8px 10px;text-align:center;font-size:10px;color:#6b7280;text-transform:uppercase;width:140px;border-right:1px solid #374151">Situation</th>
            <th style="padding:8px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase;border-right:1px solid #374151">Detected Signal</th>
            <th style="padding:8px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase;border-right:1px solid #374151">Evidence & Quote</th>
            <th style="padding:8px;text-align:left;font-size:10px;color:#6b7280;text-transform:uppercase">Source</th>
          </tr>
        </thead>
        <tbody>{"".join(sig_rows_html) if sig_rows_html else no_sig}</tbody>
      </table>
    </div>"""


def build_excluded_section(false_signals: list[tuple]) -> str:
    if not false_signals: return ""
    rows_html = ""
    for company, verified_score, reason in false_signals:
        s1 = company.get("_score_int", "?")
        rows_html += (
            f'<tr style="border-bottom:1px solid #1e293b">'
            f'<td style="padding:8px 12px;color:#6b7280;font-size:12px">{company["company"]}</td>'
            f'<td style="padding:8px 12px;color:#6b7280;font-size:12px;text-align:center">{s1}/40</td>'
            f'<td style="padding:8px 12px;color:#ef4444;font-size:12px;text-align:center">{verified_score}/40</td>'
            f'<td style="padding:8px 12px;color:#6b7280;font-size:11px">{reason}</td>'
            f'</tr>'
        )
    return f"""
    <details style="margin-bottom:24px">
      <summary style="cursor:pointer;background:#1a2535;border:1px solid #374151;border-radius:8px;
                      padding:10px 16px;color:#6b7280;font-size:12px;list-style:none">
        ▶ Excluded — {len(false_signals)} false signal(s) replaced by reserve companies
        <span style="font-size:10px;margin-left:8px;color:#4b5563">(click to expand audit trail)</span>
      </summary>
      <div style="background:#111827;border:1px solid #374151;border-top:none;border-radius:0 0 8px 8px;overflow:hidden">
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead><tr style="background:#0f172a">
            <th style="padding:8px 12px;text-align:left;color:#4b5563;font-size:10px;text-transform:uppercase">Company</th>
            <th style="padding:8px 12px;text-align:center;color:#4b5563;font-size:10px;text-transform:uppercase">S1 Score</th>
            <th style="padding:8px 12px;text-align:center;color:#4b5563;font-size:10px;text-transform:uppercase">Verified</th>
            <th style="padding:8px 12px;text-align:left;color:#4b5563;font-size:10px;text-transform:uppercase">Substitution Reason</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </details>"""


REPORT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>XIMPAX Deep Scan - {date}</title></head>
<body style="margin:0;padding:0;background:#0f172a;font-family:Arial,sans-serif;color:#e5e7eb">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;padding:24px 0">
<tr><td>
<table width="760" align="center" cellpadding="0" cellspacing="0" style="max-width:760px;margin:0 auto">

  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:12px 12px 0 0;
                 padding:28px 32px;border-bottom:3px solid #2563eb">
    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px">XIMPAX Intelligence Engine</div>
    <h1 style="margin:0;font-size:22px;color:#f9fafb;font-weight:800">Weekly Deep Scan Report</h1>
    <p style="margin:8px 0 0;color:#9ca3af;font-size:13px">Stage 2 · {date} · {num_companies} Companies · {substitutions_note} · Evidence cutoff: {cutoff}</p>
  </td></tr>

  <tr><td style="background:#1f2937;padding:16px 32px;border-bottom:1px solid #374151">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td style="text-align:center;padding:8px 0;border-right:1px solid #374151">
        <div style="font-size:26px;font-weight:800;color:#60a5fa">{num_companies}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">In Report</div>
      </td>
      <td style="text-align:center;padding:8px 0;border-right:1px solid #374151">
        <div style="font-size:26px;font-weight:800;color:#86efac">{confirmed}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Confirmed</div>
      </td>
      <td style="text-align:center;padding:8px 0;border-right:1px solid #374151">
        <div style="font-size:26px;font-weight:800;color:#93c5fd">{likely}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Likely</div>
      </td>
      <td style="text-align:center;padding:8px 0;border-right:1px solid #374151">
        <div style="font-size:26px;font-weight:800;color:#fcd34d">{clean_signals}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Clean Signals</div>
      </td>
      <td style="text-align:center;padding:8px 0">
        <div style="font-size:26px;font-weight:800;color:#f59e0b">{num_substituted}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Substituted</div>
      </td>
    </tr></table>
  </td></tr>

  <tr><td style="background:#1a2535;padding:10px 32px;border-bottom:2px solid #374151;font-size:10px;color:#6b7280">
    <b style="color:#94a3b8">Legend:</b> &nbsp;
    <span style="background:#7f1d1d;color:#fca5a5;padding:1px 6px;border-radius:4px;font-weight:700">STALE</span> evidence older than {cutoff} &nbsp;|&nbsp;
    <span style="background:#78350f;color:#fcd34d;padding:1px 6px;border-radius:4px;font-weight:700">EXCL.SOURCE</span> company website / IR page &nbsp;|&nbsp;
    <span style="background:#1e293b;color:#64748b;padding:1px 6px;border-radius:4px;font-weight:700">GENERIC</span> possible industry article (advisory) &nbsp;|&nbsp;
    <span style="background:#1e3a5f;color:#93c5fd;padding:1px 6px;border-radius:4px;font-weight:700">verified</span> grounding API URL &nbsp;|&nbsp;
    <span style="background:#374151;color:#9ca3af;padding:1px 6px;border-radius:4px;font-weight:700">unverified</span> AI-suggested URL — check before use &nbsp;|&nbsp;
    <span style="border:1px solid #f59e0b;color:#f59e0b;padding:1px 5px;border-radius:4px;font-weight:700">SUBST.</span> replaced false signal
  </td></tr>

  <tr><td style="background:#111827;padding:20px 24px">{company_sections}</td></tr>

  <tr><td style="background:#0f172a;border-radius:0 0 12px 12px;padding:16px 32px;
                 border-top:1px solid #374151;text-align:center;color:#4b5563;font-size:11px">
    XIMPAX Intelligence Engine · Gemini 2.0 Flash + Google Search<br>
    Evidence window: {cutoff} → {date} · Sorted by Stage 1 score · Substitution: verified&lt;{sub_min} | ratio&lt;{sub_ratio} | CallA&lt;{call_a_min}chars
  </td></tr>

</table></td></tr></table>
</body></html>"""


def build_report(final_companies: list[tuple], false_signals: list[tuple]) -> str:
    final_companies = sorted(
        final_companies,
        key=lambda x: x[0].get("_score_int", 0),
        reverse=True,
    )

    confirmed = likely = clean_signals = 0
    sections  = []
    for rank, (company, rows, is_sub, replaced, sub_reason) in enumerate(final_companies, 1):
        cname = company.get("company", "")
        for row in rows:
            s = row["situation_status"].lower()
            q = signal_quality(row, cname)
            if "confirmed" in s: confirmed += 1
            elif "likely"  in s: likely    += 1
            if (q["quality"] == "clean"
                    and "no signals" not in row.get("detected_signal","").lower()
                    and row.get("detected_signal","")):
                clean_signals += 1
        sections.append(build_company_section(company, rows, rank, is_sub, replaced, sub_reason))

    excluded_html = build_excluded_section(false_signals)
    num_sub       = len(false_signals)
    sub_note      = f"{num_sub} substitution(s)" if num_sub else "no substitutions"

    return REPORT_HTML.format(
        date=TODAY,
        cutoff=CUTOFF_STR,
        num_companies=len(final_companies),
        confirmed=confirmed,
        likely=likely,
        clean_signals=clean_signals,
        num_substituted=num_sub,
        substitutions_note=sub_note,
        sub_min=SUBSTITUTION_VERIFIED_MIN,
        sub_ratio=f"{int(SUBSTITUTION_RATIO*100)}%",
        call_a_min=CALL_A_MIN_CHARS,
        company_sections=excluded_html + "\n".join(sections),
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
    log.info(f"Email sent to {to_addr}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    primary, reserve = load_stage2_input()
    log.info(
        f"Stage 2: {len(primary)} primary | {len(reserve)} reserve | "
        f"Thresholds: verified<{SUBSTITUTION_VERIFIED_MIN} | "
        f"ratio<{int(SUBSTITUTION_RATIO*100)}% | CallA<{CALL_A_MIN_CHARS}chars"
    )

    raw_all:       dict = {}
    false_signals: list = []   # (company, verified_score, reason)
    final_results: list = []   # (company, rows, is_substitute, replaced_or_None, reason)
    reserve_queue: list = list(reserve)

    # ── Phase 1: Scan primaries ────────────────────────────────────────────────
    log.info("\n=== PHASE 1: Primary companies ===")
    for i, company in enumerate(primary):
        cname = company["company"]
        log.info(f"[P{i+1}/{len(primary)}] {cname} (S1={company.get('_score_int','?')}/40)")
        try:
            sleep_after = not (i == len(primary) - 1 and not reserve_queue)
            rows, research_text, table_text, call_a_chars = deep_scan_company(
                company, sleep_after=sleep_after
            )
            raw_all[cname] = {"research": research_text, "table": table_text}

            verified = recalculate_score(rows, cname)
            log.info(
                f"  -> Verified={verified}/40 | S1={company.get('_score_int','?')}/40 | "
                f"CallA={call_a_chars}chars"
            )

            triggered, reason = is_false_signal(company, rows, verified, call_a_chars)
            if triggered:
                false_signals.append((company, verified, reason))
                log.info(f"  -> Reserve pool remaining: {len(reserve_queue)}")
            else:
                final_results.append((company, rows, False, None, ""))

        except Exception as e:
            log.warning(f"  -> Scan failed: {e}")
            final_results.append((company, [], False, None, ""))

    # ── Phase 2: Substitute false signals ─────────────────────────────────────
    if false_signals:
        log.info(f"\n=== PHASE 2: Substituting {len(false_signals)} false signal(s) ===")
        for orig_company, orig_verified, orig_reason in false_signals:
            if not reserve_queue:
                log.warning("Reserve pool exhausted — keeping original with empty rows")
                final_results.append((orig_company, [], False, None, ""))
                continue

            # ── First substitute attempt ──────────────────────────────────────
            substitute = reserve_queue.pop(0)
            log.info(
                f"  {orig_company['company']} (verified={orig_verified}) "
                f"→ {substitute['company']} (S1={substitute.get('_score_int','?')}/40)"
            )
            try:
                sleep_after = bool(reserve_queue)
                rows, research_text, table_text, call_a_chars = deep_scan_company(
                    substitute, sleep_after=sleep_after
                )
                raw_all[substitute["company"]] = {"research": research_text, "table": table_text}
                verified = recalculate_score(rows, substitute["company"])
                log.info(f"  -> Substitute verified={verified}/40 | CallA={call_a_chars}chars")

                # ── If substitute itself scores 0, try ONE more reserve ───────
                # (A 0-score substitute is worse than nothing — allow one further pull)
                if verified == 0 and reserve_queue:
                    log.warning(
                        f"  -> Substitute {substitute['company']} also scored 0 — "
                        f"trying one more reserve (no further chain after this)"
                    )
                    second_sub = reserve_queue.pop(0)
                    log.info(
                        f"  {substitute['company']} → {second_sub['company']} "
                        f"(S1={second_sub.get('_score_int','?')}/40)"
                    )
                    try:
                        sleep_after2 = bool(reserve_queue)
                        rows2, rt2, tt2, ca2 = deep_scan_company(
                            second_sub, sleep_after=sleep_after2
                        )
                        raw_all[second_sub["company"]] = {"research": rt2, "table": tt2}
                        verified2 = recalculate_score(rows2, second_sub["company"])
                        log.info(f"  -> 2nd substitute verified={verified2}/40 | CallA={ca2}chars")
                        if verified2 > 0:
                            log.info(f"  -> Using 2nd substitute {second_sub['company']} (verified={verified2})")
                            final_results.append((second_sub, rows2, True, orig_company, orig_reason))
                        else:
                            log.warning(f"  -> Both substitutes scored 0. Using first: {substitute['company']}")
                            final_results.append((substitute, rows, True, orig_company, orig_reason))
                    except Exception as e2:
                        log.warning(f"  -> 2nd substitute scan failed: {e2} — using first")
                        final_results.append((substitute, rows, True, orig_company, orig_reason))
                else:
                    if verified == 0 and not reserve_queue:
                        log.warning(f"  -> Substitute scored 0, reserve pool exhausted. Including with warning.")
                    final_results.append((substitute, rows, True, orig_company, orig_reason))

            except Exception as e:
                log.warning(f"  -> Substitute scan failed: {e}")
                final_results.append((substitute, [], True, orig_company, orig_reason))

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info(f"\n=== FINAL SUMMARY ===")
    log.info(f"Total scans run: {len(primary) + len(false_signals)}")
    log.info(f"False signals removed: {len(false_signals)}")
    log.info(f"Report companies: {len(final_results)}")
    for company, rows, is_sub, replaced, reason in sorted(
        final_results, key=lambda x: x[0].get("_score_int", 0), reverse=True
    ):
        verified = recalculate_score(rows, company["company"])
        tag = f" [SUBST from {replaced['company']}]" if is_sub and replaced else ""
        log.info(f"  {company['company']:35s} S1={company.get('_score_int','?'):>2} V={verified:>2}{tag}")

    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"stage2_raw_{ts}.json"
    raw_path.write_text(json.dumps(raw_all, ensure_ascii=False, indent=2))

    html      = build_report(final_results, false_signals)
    html_path = OUTPUT_DIR / f"stage2_report_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Stage 2 report -> {html_path}")

    try:
        send_email(html, html_path)
    except Exception as e:
        log.error(f"Email failed: {e}")
        raise

    return html_path


if __name__ == "__main__":
    main()
