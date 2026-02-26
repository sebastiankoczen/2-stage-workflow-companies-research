"""
XIMPAX Intelligence Engine — Stage 2
Two-call architecture per company:
  Call A → Gemini + Google Search → deep prose research (real URLs, real quotes)
  Call B → Gemini (no tools) → formats findings into 4-column signal table

Then builds HTML email report and sends via Gmail.
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

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
CONFIG_DIR  = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
OUTPUT_DIR  = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

GEMINI_MODEL = "gemini-2.0-flash"
SLEEP_BETWEEN = 10   # seconds between companies

CUTOFF_DATE = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
TODAY       = datetime.utcnow().strftime("%Y-%m-%d")


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_stage2_input() -> list[dict]:
    path = OUTPUT_DIR / "stage2_input.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 input missing: {path}")
    data = json.loads(path.read_text())
    data.sort(key=lambda r: r.get("_score_int", 0), reverse=True)
    log.info(f"Stage 2 input: {len(data)} companies (sorted by score):")
    for r in data:
        log.info(f"  → {r['company']} | {r.get('_score_int','?')}/40")
    return data


# ── CALL A: Research prompt ────────────────────────────────────────────────────
def build_research_prompt(company: dict) -> str:
    instructions    = load_file(CONFIG_DIR / "instructions.txt")
    icp_blueprint   = load_file(CONFIG_DIR / "icp_blueprint.txt")

    return f"""
You are a senior supply chain market intelligence analyst. Today is {TODAY}.

Deep-scan this company for supply chain execution signals:
Company: {company['company']}
Country: {company['hq_country']}
Industry: {company['industry']}
Revenue: {company['revenue']}
Stage 1 preliminary findings: {company.get('situations', 'unknown')} | Score: {company.get('priority_score', 'unknown')}

You MUST research ALL FOUR situations independently by running these searches:

SITUATION 1 — RESOURCE CONSTRAINTS
Run: "{company['company']} supply chain procurement staffing shortage capacity bandwidth 2024 2025"
Run: "{company['company']} CPO supply chain director departure vacancy urgent hire 2024 2025"
Run: "{company['company']} ERP IBP S&OP implementation delayed resource gap 2024 2025"

SITUATION 2 — MARGIN PRESSURE  
Run: "{company['company']} EBITDA gross margin decline cost reduction restructuring 2024 2025"
Run: "{company['company']} profit warning savings program procurement efficiency 2024 2025"
Run: "{company['company']} layoffs plant closure footprint SKU reduction 2024 2025"

SITUATION 3 — SIGNIFICANT GROWTH
Run: "{company['company']} acquisition merger new plant capacity expansion 2024 2025"
Run: "{company['company']} revenue growth supply chain scaling backlog new market 2024 2025"

SITUATION 4 — SUPPLY CHAIN DISRUPTION
Run: "{company['company']} supply disruption shortage production halt 2024 2025"
Run: "{company['company']} recall quality crisis supplier failure logistics disruption 2024 2025"

SCORING RULES (apply strictly):
STRONG signal = +2 pts | MEDIUM signal = +1 pt | Cap per situation = 10 pts
CONFIRMED = 10+ pts | LIKELY = 6-9 pts | UNCLEAR = 3-5 pts | NOT PRESENT = 0-2 pts

EVIDENCE RULES:
- Only evidence dated after {CUTOFF_DATE}. Reject anything older.
- Preferred sources: Reuters, Bloomberg, FT, earnings transcripts, regulatory filings, Seeking Alpha
- FORBIDDEN sources: company website, investor relations pages, vendor blogs
- Every piece of evidence MUST have a real URL — if you cannot find one, write "no evidence found"
- Include verbatim quotes under 25 words

=== SITUATION SIGNAL DEFINITIONS ===
{instructions}

=== ICP BLUEPRINT ===
{icp_blueprint}

OUTPUT FORMAT — write one block per situation:

SITUATION 1 — RESOURCE CONSTRAINTS: [CONFIRMED/LIKELY/UNCLEAR/NOT PRESENT] ([X] pts)
Signal: [exact signal name from framework] | Strength: STRONG/MEDIUM | Weight: +2/+1
Quote: "[verbatim quote, max 25 words]"
Date: YYYY-MM-DD | Source: [Publication] | URL: [full https:// URL]
[repeat for each signal, or write "No evidence found above threshold"]

SITUATION 2 — MARGIN PRESSURE: [classification] ([X] pts)
[same format]

SITUATION 3 — SIGNIFICANT GROWTH: [classification] ([X] pts)
[same format]

SITUATION 4 — SUPPLY CHAIN DISRUPTION: [classification] ([X] pts)
[same format]

TOTAL SCORE: RC:[n] | MP:[n] | SG:[n] | SCD:[n] = [total]/40
DOMINANT SITUATION: [the highest-scoring one]
"""


# ── CALL B: Table formatting prompt ───────────────────────────────────────────
def build_format_prompt_s2(company: dict, research_text: str) -> str:
    prompt_template = load_file(PROMPTS_DIR / "prompt_2.txt")

    return f"""
You are a data formatting engine. Convert the research findings below into the exact
markdown table format specified. Do NOT do additional research. Do NOT change scores,
quotes, or URLs. Just reformat the research into the table.

=== RESEARCH FINDINGS ===
{research_text}

=== TABLE FORMAT REQUIRED ===
{prompt_template}

CRITICAL RULES:
1. Output ONLY the markdown table — no text before or after
2. Every row must start with | and end with |
3. The table has exactly 4 columns: Situation Status | Detected Signal | Evidence & Quote | Source & URL
4. One row per signal detected
5. Source & URL column: ONLY use URLs that appear verbatim in the research findings above
   — do NOT invent, guess, or modify any URL
6. If research shows "no evidence found" for a situation, write one row with "No signals detected"
"""


# ── Gemini API calls ──────────────────────────────────────────────────────────
def gemini_research(prompt: str, company_name: str) -> str:
    """Call A: research with Google Search grounding."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    log.info(f"  → Call A: researching {company_name} with Google Search …")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
            max_output_tokens=16000,
        ),
    )
    text = response.text or ""
    log.info(f"  → Call A: {len(text)} chars returned")
    return text


def gemini_format(research_text: str, company: dict) -> str:
    """Call B: format prose research into 4-column table (no tools)."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    log.info(f"  → Call B: formatting {company['company']} into table …")

    format_prompt = build_format_prompt_s2(company, research_text)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=format_prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a data formatting engine. Output ONLY the markdown table. "
                "Start with the | character of the header row. "
                "Every row must start and end with |. "
                "No text before or after the table. "
                "Never invent or modify URLs — only use URLs present in the input."
            ),
            temperature=0.0,
            max_output_tokens=8000,
        ),
    )
    text = response.text or ""
    log.info(f"  → Call B: {len(text)} chars returned")
    return text


# ── Parse Stage 2 table ───────────────────────────────────────────────────────
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


# ── Score recalculation ───────────────────────────────────────────────────────
def recalculate_score(rows: list[dict]) -> int:
    """Sum STRONG(+2) and MEDIUM(+1) signals from parsed rows, cap 10 per situation."""
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
        w = 2 if "strong" in sig else (1 if "medium" in sig else 0)
        for k in sit_scores:
            if k in current:
                sit_scores[k] = min(10, sit_scores[k] + w)
                break
    return sum(sit_scores.values())


# ── HTML report builder ───────────────────────────────────────────────────────
SIGNAL_STYLES = {
    "confirmed":   ("#86efac", "#14532d", "✅ CONFIRMED"),
    "likely":      ("#93c5fd", "#1e3a5f", "🔵 LIKELY"),
    "unclear":     ("#fcd34d", "#422006", "⚠️ UNCLEAR"),
    "not present": ("#9ca3af", "#1f2937", "⬜ NOT PRESENT"),
}

def situation_style(s: str):
    t = s.lower()
    for k, v in SIGNAL_STYLES.items():
        if k in t:
            return v
    return ("#d1d5db", "#1f2937", s[:40])


def make_source_link(src: str) -> str:
    urls  = re.findall(r"https?://[^\s<>\"'\]]+", src)
    label = re.sub(r"https?://[^\s<>\"'\]]+", "", src).strip(" —-–[]")
    if urls:
        return f'<a href="{urls[0]}" target="_blank" style="color:#60a5fa">{label or "🔗 Source"}</a>'
    # No URL — show plain text but flag it
    if src.strip() and src.strip() not in ("-", "—"):
        return f'<span style="color:#6b7280;font-size:10px">{src[:80]}</span>'
    return ""


def build_company_section(company: dict, rows: list[dict], rank: int) -> str:
    score     = recalculate_score(rows) if rows else company.get("_score_int", 0)
    bar_pct   = min(100, int(score / 40 * 100))
    dom_color = "#16a34a" if any("confirmed" in r["situation_status"].lower() for r in rows) else "#2563eb"

    # Group by situation block
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
            sit_td = (
                f'<td style="background:{bg};color:{tc};font-weight:700;font-size:11px;'
                f'white-space:nowrap;padding:10px 8px;border-right:1px solid #374151">'
                f'{label}<br><small style="font-size:9px;font-weight:400;opacity:.8">{sit}</small></td>'
            ) if first else '<td style="background:#111827;border-right:1px solid #374151"></td>'
            first = False
            sig_rows_html.append(f"""
            <tr style="border-bottom:1px solid #374151">
              {sit_td}
              <td style="padding:10px 8px;font-size:12px;color:#d1d5db;border-right:1px solid #374151">{r['detected_signal']}</td>
              <td style="padding:10px 8px;font-size:11px;font-style:italic;color:#9ca3af;border-right:1px solid #374151">{r['evidence']}</td>
              <td style="padding:10px 8px;font-size:11px">{make_source_link(r['source_url'])}</td>
            </tr>""")

    no_sig = ('<tr><td colspan="4" style="padding:14px;color:#6b7280;text-align:center;'
              'font-style:italic">No signals detected</td></tr>')

    return f"""
    <div style="background:#1f2937;border-radius:10px;margin-bottom:24px;overflow:hidden;border:1px solid #374151">
      <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-left:4px solid {dom_color};
                  padding:16px 20px;display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">#{rank} Priority Lead</div>
          <h2 style="font-size:18px;margin:0;color:#f9fafb;font-weight:700">{company['company']}</h2>
          <p style="font-size:12px;color:#9ca3af;margin:5px 0 0">{company['industry']} &nbsp;·&nbsp; {company['hq_country']} &nbsp;·&nbsp; {company['revenue']}</p>
        </div>
        <div style="text-align:right;min-width:120px">
          <div style="font-size:30px;font-weight:800;color:#60a5fa">{score}<small style="font-size:14px;color:#6b7280">/40</small></div>
          <div style="background:rgba(255,255,255,.1);border-radius:4px;height:6px;width:110px;margin:6px 0 2px;margin-left:auto">
            <div style="background:linear-gradient(90deg,#2563eb,#7c3aed);height:6px;border-radius:4px;width:{bar_pct}%"></div>
          </div>
          <small style="font-size:10px;color:#6b7280">Stage 2 Score</small>
        </div>
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
    <p style="margin:8px 0 0;color:#9ca3af;font-size:13px">Stage 2 · {date} · {num_companies} Companies · Live Web Research</p>
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
        <div style="font-size:26px;font-weight:800;color:#fcd34d">{total_signals}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Signals</div>
      </td>
    </tr></table>
  </td></tr>

  <tr><td style="background:#111827;padding:20px 24px">{company_sections}</td></tr>

  <tr><td style="background:#0f172a;border-radius:0 0 12px 12px;padding:16px 32px;
                 border-top:1px solid #374151;text-align:center;color:#4b5563;font-size:11px">
    XIMPAX Intelligence Engine · Automated Weekly Report · Gemini AI + Google Search
  </td></tr>

</table></td></tr></table>
</body></html>"""


def build_report(companies_with_rows: list[tuple]) -> str:
    confirmed = likely = total_signals = 0
    sections  = []

    for rank, (company, rows) in enumerate(companies_with_rows, 1):
        for row in rows:
            s = row["situation_status"].lower()
            if "confirmed" in s: confirmed += 1
            elif "likely"  in s: likely    += 1
            if row["detected_signal"] and "no signals" not in row["detected_signal"].lower():
                total_signals += 1
        sections.append(build_company_section(company, rows, rank))

    return REPORT_HTML.format(
        date=datetime.utcnow().strftime("%B %d, %Y"),
        num_companies=len(companies_with_rows),
        confirmed=confirmed, likely=likely,
        total_signals=total_signals,
        company_sections="\n".join(sections),
    )


# ── Email ─────────────────────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    companies = load_stage2_input()
    log.info(f"Stage 2: scanning {len(companies)} companies")

    companies_with_rows: list[tuple] = []
    raw_all: dict = {}

    for i, company in enumerate(companies):
        try:
            # Call A: research with live Google Search
            research_text = gemini_research(
                build_research_prompt(company), company["company"]
            )
            raw_all[company["company"]] = {"research": research_text}

            # Small pause between the two calls
            time.sleep(5)

            # Call B: format research into table
            table_text = gemini_format(research_text, company)
            raw_all[company["company"]]["table"] = table_text

            rows = parse_stage2_table(table_text)
            log.info(f"  → {len(rows)} signal rows for {company['company']}")
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
