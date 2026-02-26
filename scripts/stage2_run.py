"""
XIMPAX Intelligence Engine — Stage 2
Deep-scans the Top 10 companies from Stage 1 via Gemini + Prompt 2,
generates a polished HTML report, and emails it via Gmail.
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
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = ROOT / "config"
PROMPTS_DIR  = ROOT / "prompts"
OUTPUT_DIR   = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
SLEEP_BETWEEN = 8        # seconds between company calls (rate limit guard)


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_stage2_input() -> list[dict]:
    path = OUTPUT_DIR / "stage2_input.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 input not found at {path}. Run stage1_run.py first.")
    return json.loads(path.read_text())


def build_prompt(company: dict) -> str:
    company_profile = load_file(CONFIG_DIR / "company_profile.txt")
    icp_blueprint   = load_file(CONFIG_DIR / "icp_blueprint.txt")
    instructions    = load_file(CONFIG_DIR / "instructions.txt")
    prompt_template = load_file(PROMPTS_DIR / "prompt_2.txt")

    company_context = f"""
Company to deep-scan:
- Name: {company['company']}
- HQ Country: {company['hq_country']}
- Industry: {company['industry']}
- Revenue: {company['revenue']}
- Stage 1 Situations Detected: {company['situations']}
- Stage 1 Priority Score: {company['priority_score']}
- Stage 1 Key Evidence: {company['key_evidence']}

MANDATORY SEARCH INSTRUCTIONS FOR THIS COMPANY:
You MUST run separate Google searches for ALL 4 situations before writing any row.
Use these exact search queries (replace [COMPANY] with the company name):

Search 1 — Resource Constraints:
  "[COMPANY] supply chain staffing shortage capacity gap bandwidth 2024 2025"
  "[COMPANY] SC procurement leadership departure vacancy urgent hire 2024 2025"

Search 2 — Margin Pressure:
  "[COMPANY] EBITDA margin decline cost reduction program restructuring 2024 2025"
  "[COMPANY] profit warning cost savings target procurement efficiency 2024 2025"

Search 3 — Significant Growth:
  "[COMPANY] acquisition M&A expansion new facility plant investment 2024 2025"
  "[COMPANY] revenue growth supply chain scaling capacity 2024 2025"

Search 4 — Supply Chain Disruption:
  "[COMPANY] supply disruption shortage recall logistics failure 2024 2025"
  "[COMPANY] production shutdown supplier failure inventory shock 2024 2025"

Only include evidence from the last 12 months. Reject any source older than 12 months.
Do NOT use company websites or investor relations pages as sources.
"""

    return f"""
=== FIRM PROFILE DOCUMENT ===
{company_profile}

=== ICP BLUEPRINT ===
{icp_blueprint}

=== SITUATION DETECTION FRAMEWORK ===
{instructions}

=== COMPANY CONTEXT & SEARCH INSTRUCTIONS ===
{company_context}

=== DEEP SCAN TASK ===
{prompt_template}
"""


# ── Gemini call ────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are a supply chain market intelligence engine. "
    "Use your most recent knowledge to find evidence for supply chain signals. "
    "Do NOT use company websites or investor relations pages as sources. "
    "Output ONLY the markdown table — no preamble, no acknowledgement, no text before or after. "
    "Start your response directly with the | character of the header row. "
    "Every row must start AND end with a | character. "
    "Do not truncate — output all rows."
)

def call_gemini(prompt: str, company_name: str) -> str:
    api_key = os.environ["GEMINI_API_KEY"]
    client  = genai.Client(api_key=api_key)
    log.info(f"Deep scanning: {company_name} …")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=16000,
        ),
    )
    return response.text


# ── Parse Stage 2 markdown table ───────────────────────────────────────────────
def parse_stage2_table(raw: str) -> list[dict]:
    """Parse the 4-column deep scan table from Prompt 2 output."""
    raw = re.sub(r"```[a-z]*", "", raw)
    lines = [l.strip() for l in raw.splitlines()]
    pipe_lines = [l for l in lines if l.startswith("|")]

    if not pipe_lines:
        log.warning("Stage 2: no pipe-delimited lines found in response.")
        return []

    rows = []
    header_skipped = False

    for line in pipe_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]

        # Skip separator rows
        if all(re.match(r"^[-:\s]+$", c) for c in cells if c):
            header_skipped = True
            continue

        # Skip header row
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


# ── HTML report builder ─────────────────────────────────────────────────────────
SIGNAL_COLORS = {
    "confirmed": ("#166534", "#dcfce7", "✅ CONFIRMED"),
    "likely":    ("#1e40af", "#dbeafe", "🔵 LIKELY"),
    "unclear":   ("#854d0e", "#fef9c3", "⚠️ UNCLEAR"),
    "not present": ("#64748b", "#f1f5f9", "⬜ NOT PRESENT"),
}

def situation_style(status_text: str):
    t = status_text.lower()
    for key, val in SIGNAL_COLORS.items():
        if key in t:
            return val
    return ("#333", "#f8fafc", status_text)


def make_source_link(src: str) -> str:
    urls = re.findall(r"https?://[^\s<>\"']+", src)
    label = re.sub(r"https?://[^\s<>\"']+", "", src).strip(" —-–")
    if urls:
        return f'<a href="{urls[0]}" target="_blank">{label or "🔗 Source"}</a>'
    return src[:100]


def recalculate_score_from_rows(rows: list[dict]) -> int:
    """
    Recalculate the total priority score from Stage 2 parsed signal rows.
    STRONG signals = +2, MEDIUM = +1, capped at 10 per situation.
    This is the authoritative score — replaces whatever Stage 1 estimated.
    """
    sit_scores: dict[str, int] = {
        "resource constraints": 0,
        "margin pressure": 0,
        "significant growth": 0,
        "supply chain disruption": 0,
    }
    current_sit = ""
    for row in rows:
        if row["situation_status"].strip():
            current_sit = row["situation_status"].lower()
        signal = row.get("detected_signal", "").lower()
        if "no signals" in signal or not signal:
            continue
        weight = 2 if "strong" in signal else (1 if "medium" in signal else 0)
        for sit_key in sit_scores:
            if sit_key in current_sit:
                sit_scores[sit_key] = min(10, sit_scores[sit_key] + weight)
                break
    return sum(sit_scores.values())


SIGNAL_COLORS = {
    "confirmed":   ("#86efac", "#14532d", "✅ CONFIRMED"),
    "likely":      ("#93c5fd", "#1e3a5f", "🔵 LIKELY"),
    "unclear":     ("#fcd34d", "#422006", "⚠️ UNCLEAR"),
    "not present": ("#9ca3af", "#1f2937", "⬜ NOT PRESENT"),
}

def situation_style(status_text: str):
    t = status_text.lower()
    for key, val in SIGNAL_COLORS.items():
        if key in t:
            return val
    return ("#d1d5db", "#1f2937", status_text[:40])


def build_company_section(company: dict, rows: list[dict], rank: int) -> str:
    # Score recalculated from live Stage 2 findings — not the Stage 1 estimate
    score_int = recalculate_score_from_rows(rows)
    if score_int == 0 and not rows:
        score_int = company.get("_score_int", 0)  # fallback only if totally empty

    bar_pct = min(100, int(score_int / 40 * 100))

    # Dominant colour: green if any Confirmed, blue if any Likely
    dominant_color = "#2563eb"
    for row in rows:
        if "confirmed" in row["situation_status"].lower():
            dominant_color = "#16a34a"
            break

    # Group rows by situation block
    groups: dict[str, list] = {}
    current_sit = ""
    for row in rows:
        if row["situation_status"].strip():
            current_sit = row["situation_status"]
        groups.setdefault(current_sit, []).append(row)

    signal_rows_html = []
    for sit, sig_rows in groups.items():
        txt_color, bg_color, label = situation_style(sit)
        first = True
        for sig_row in sig_rows:
            if first:
                sit_cell = (
                    f'<td style="background:{bg_color};color:{txt_color};font-weight:700;'
                    f'font-size:11px;white-space:nowrap;padding:10px 8px;'
                    f'border-right:1px solid #374151">'
                    f'{label}<br><small style="font-size:9px;font-weight:400;opacity:.8">{sit}</small></td>'
                )
                first = False
            else:
                sit_cell = '<td style="background:#111827;border-right:1px solid #374151"></td>'

            signal_rows_html.append(f"""
            <tr style="border-bottom:1px solid #374151">
              {sit_cell}
              <td style="padding:10px 8px;font-size:12px;color:#d1d5db;border-right:1px solid #374151">{sig_row['detected_signal']}</td>
              <td style="padding:10px 8px;font-size:11px;font-style:italic;color:#9ca3af;border-right:1px solid #374151">{sig_row['evidence']}</td>
              <td style="padding:10px 8px;font-size:11px">{make_source_link(sig_row['source_url'])}</td>
            </tr>""")

    no_signals = (
        '<tr><td colspan="4" style="padding:14px;color:#6b7280;text-align:center;font-style:italic">'
        'No signals parsed — check raw output in repo</td></tr>'
    )

    return f"""
    <div style="background:#1f2937;border-radius:10px;margin-bottom:24px;overflow:hidden;border:1px solid #374151">
      <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-left:4px solid {dominant_color};
                  padding:16px 20px;display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
            #{rank} Priority Lead
          </div>
          <h2 style="font-size:18px;margin:0;color:#f9fafb;font-weight:700">{company['company']}</h2>
          <p style="font-size:12px;color:#9ca3af;margin:5px 0 0">
            {company['industry']} &nbsp;·&nbsp; {company['hq_country']} &nbsp;·&nbsp; {company['revenue']}
          </p>
        </div>
        <div style="text-align:right;min-width:120px">
          <div style="font-size:30px;font-weight:800;color:#60a5fa">{score_int}<small style="font-size:14px;color:#6b7280">/40</small></div>
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
        <tbody>
          {"".join(signal_rows_html) if signal_rows_html else no_signals}
        </tbody>
      </table>
    </div>"""


REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XIMPAX Deep Scan Report — {date}</title>
</head>
<body style="margin:0;padding:0;background:#0f172a;font-family:Arial,sans-serif;color:#e5e7eb">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;padding:24px 0">
<tr><td>
<table width="700" align="center" cellpadding="0" cellspacing="0" style="max-width:700px;margin:0 auto">

  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:12px 12px 0 0;
                 padding:28px 32px;border-bottom:3px solid #2563eb">
    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px">XIMPAX Intelligence Engine</div>
    <h1 style="margin:0;font-size:22px;color:#f9fafb;font-weight:800">🎯 Weekly Deep Scan Report</h1>
    <p style="margin:8px 0 0;color:#9ca3af;font-size:13px">
      Stage 2 Analysis &nbsp;·&nbsp; {date} &nbsp;·&nbsp; Top {num_companies} Companies Validated
    </p>
  </td></tr>

  <tr><td style="background:#1f2937;padding:16px 32px;border-bottom:1px solid #374151">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td style="text-align:center;padding:8px 0;border-right:1px solid #374151">
        <div style="font-size:28px;font-weight:800;color:#60a5fa">{num_companies}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Scanned</div>
      </td>
      <td style="text-align:center;padding:8px 0;border-right:1px solid #374151">
        <div style="font-size:28px;font-weight:800;color:#86efac">{confirmed}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Confirmed</div>
      </td>
      <td style="text-align:center;padding:8px 0;border-right:1px solid #374151">
        <div style="font-size:28px;font-weight:800;color:#93c5fd">{likely}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Likely</div>
      </td>
      <td style="text-align:center;padding:8px 0">
        <div style="font-size:28px;font-weight:800;color:#fcd34d">{total_signals}</div>
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Signals</div>
      </td>
    </tr></table>
  </td></tr>

  <tr><td style="background:#111827;padding:20px 24px">
    {company_sections}
  </td></tr>

  <tr><td style="background:#0f172a;border-radius:0 0 12px 12px;padding:16px 32px;
                 border-top:1px solid #374151;text-align:center;color:#4b5563;font-size:11px">
    XIMPAX Intelligence Engine &nbsp;·&nbsp; Automated Weekly Report &nbsp;·&nbsp; Powered by Gemini AI + Google Search
  </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""


def build_full_report(companies_with_rows: list[tuple]) -> str:
    date_str      = datetime.utcnow().strftime("%B %d, %Y")
    confirmed     = 0
    likely        = 0
    total_signals = 0

    sections = []
    for rank, (company, rows) in enumerate(companies_with_rows, start=1):
        for row in rows:
            sit = row["situation_status"].lower()
            if "confirmed" in sit:   confirmed += 1
            elif "likely" in sit:    likely    += 1
            if row["detected_signal"] and "no signals" not in row["detected_signal"].lower():
                total_signals += 1
        sections.append(build_company_section(company, rows, rank))

    return REPORT_TEMPLATE.format(
        date=date_str,
        num_companies=len(companies_with_rows),
        confirmed=confirmed,
        likely=likely,
        total_signals=total_signals,
        company_sections="\n".join(sections),
    )



# ── Email sender ────────────────────────────────────────────────────────────────
def send_email(html_content: str, attachment_path: Path):
    smtp_user     = os.environ["GMAIL_ADDRESS"]
    smtp_password = os.environ["GMAIL_APP_PASSWORD"]
    to_address    = os.environ["RECIPIENT_EMAIL"]

    subject = f"XIMPAX Weekly Intelligence Report — {datetime.utcnow().strftime('%d %b %Y')}"

    msg = MIMEMultipart("mixed")
    msg["From"]    = smtp_user
    msg["To"]      = to_address
    msg["Subject"] = subject

    # HTML body
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    # Attachment
    with open(attachment_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="{attachment_path.name}"',
    )
    msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_address, msg.as_string())

    log.info(f"Email sent to {to_address}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    companies = load_stage2_input()
    log.info(f"Stage 2: deep scanning {len(companies)} companies")

    companies_with_rows = []
    raw_all = {}

    for i, company in enumerate(companies):
        try:
            prompt = build_prompt(company)
            raw = call_gemini(prompt, company["company"])
            raw_all[company["company"]] = raw
            rows = parse_stage2_table(raw)
            log.info(f"  → {len(rows)} signal rows parsed for {company['company']}")
            companies_with_rows.append((company, rows))
        except Exception as e:
            log.warning(f"Failed for {company['company']}: {e}")
            companies_with_rows.append((company, []))

        if i < len(companies) - 1:
            time.sleep(SLEEP_BETWEEN)

    # Save raw
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"stage2_raw_{ts}.json"
    raw_path.write_text(json.dumps(raw_all, ensure_ascii=False, indent=2))

    # Build HTML report
    html = build_full_report(companies_with_rows)
    html_path = OUTPUT_DIR / f"stage2_report_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Stage 2 HTML report → {html_path}")

    # Send email
    try:
        send_email(html, html_path)
        log.info("✅ Email sent successfully")
    except Exception as e:
        log.error(f"Email failed: {e}")
        raise

    return html_path


if __name__ == "__main__":
    main()
