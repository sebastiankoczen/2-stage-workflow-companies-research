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
"""

    return f"""
=== FIRM PROFILE DOCUMENT ===
{company_profile}

=== ICP BLUEPRINT ===
{icp_blueprint}

=== SITUATION DETECTION FRAMEWORK ===
{instructions}

=== COMPANY CONTEXT (from Stage 1) ===
{company_context}

=== DEEP SCAN TASK ===
{prompt_template}
"""


# ── Gemini call ────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are a supply chain market intelligence engine. "
    "When asked to produce a table, output ONLY the markdown table with no preamble, "
    "no acknowledgement, no explanation before or after. "
    "Start your response directly with the | character of the first table row. "
    "Every row must start AND end with a | character. "
    "Do not truncate the table — include all rows."
)

def call_gemini(prompt: str, company_name: str) -> str:
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
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


def build_company_section(company: dict, rows: list[dict]) -> str:
    score_int = company.get("_score_int", 0)
    bar_pct = min(100, int(score_int / 40 * 100))

    # Group rows by situation
    groups: dict[str, list] = {}
    current_situation = ""
    for row in rows:
        if row["situation_status"].strip():
            current_situation = row["situation_status"]
        groups.setdefault(current_situation, []).append(row)

    signal_rows_html = []
    for sit, sig_rows in groups.items():
        color, bg, label = situation_style(sit)
        first = True
        for sig_row in sig_rows:
            sit_cell = f'<td style="background:{bg};color:{color};font-weight:600;font-size:11px;white-space:nowrap;padding:8px">{label}<br><small style="font-size:10px;font-weight:normal">{sit}</small></td>' if first else '<td style="background:#fafafa"></td>'
            first = False
            signal_rows_html.append(f"""
            <tr>
              {sit_cell}
              <td style="padding:8px;font-size:12px">{sig_row['detected_signal']}</td>
              <td style="padding:8px;font-size:11px;font-style:italic;color:#444">{sig_row['evidence']}</td>
              <td style="padding:8px;font-size:11px">{make_source_link(sig_row['source_url'])}</td>
            </tr>""")

    return f"""
    <div style="background:#fff;border-radius:10px;margin-bottom:28px;box-shadow:0 2px 8px rgba(0,0,0,.08);overflow:hidden">
      <div style="background:linear-gradient(135deg,#1a1a2e,#2563eb);color:#fff;padding:16px 20px;display:flex;justify-content:space-between;align-items:center">
        <div>
          <h2 style="font-size:17px;margin:0">{company['company']}</h2>
          <p style="font-size:12px;opacity:.85;margin:4px 0 0">{company['industry']} &nbsp;|&nbsp; {company['hq_country']} &nbsp;|&nbsp; {company['revenue']}</p>
        </div>
        <div style="text-align:right">
          <div style="font-size:26px;font-weight:700">{score_int}<small style="font-size:13px">/40</small></div>
          <div style="background:rgba(255,255,255,.2);border-radius:4px;height:6px;width:120px;margin-top:4px">
            <div style="background:#7dd3fc;height:6px;border-radius:4px;width:{bar_pct}%"></div>
          </div>
          <small style="font-size:10px;opacity:.7">Priority Score</small>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;font-family:Arial,sans-serif">
        <thead>
          <tr style="background:#f1f5f9">
            <th style="padding:8px;text-align:left;font-size:11px;color:#555;text-transform:uppercase;width:160px">Situation</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#555;text-transform:uppercase">Detected Signal</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#555;text-transform:uppercase">Evidence & Quote</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#555;text-transform:uppercase;width:150px">Source</th>
          </tr>
        </thead>
        <tbody>
          {"".join(signal_rows_html) or '<tr><td colspan="4" style="padding:12px;color:#999;text-align:center">No signals parsed — check raw output</td></tr>'}
        </tbody>
      </table>
    </div>"""


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XIMPAX Deep Scan Report — {date}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: Arial, sans-serif; background: #f4f6f9; color: #1a1a2e; margin: 0; padding: 24px; }}
  .header {{ background: linear-gradient(135deg,#1a1a2e,#1e40af); color: #fff; padding: 28px 32px; border-radius: 12px; margin-bottom: 28px; }}
  .header h1 {{ font-size: 24px; margin: 0 0 6px; }}
  .header p {{ margin: 0; opacity: .8; font-size: 13px; }}
  .stats {{ display: flex; gap: 14px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat {{ background: #fff; border-radius: 8px; padding: 14px 20px; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  .stat .v {{ font-size: 30px; font-weight: 700; color: #2563eb; }}
  .stat .l {{ font-size: 11px; color: #888; text-transform: uppercase; }}
  a {{ color: #2563eb; }}
  .footer {{ text-align:center; color:#888; font-size:11px; margin-top:24px; padding-top:16px; border-top:1px solid #e2e8f0; }}
</style>
</head>
<body>
<div class="header">
  <h1>🎯 XIMPAX Weekly Intelligence — Deep Scan Report</h1>
  <p>Stage 2 Analysis | {date} | Top {num_companies} Companies Validated</p>
</div>

<div class="stats">
  <div class="stat"><div class="v">{num_companies}</div><div class="l">Companies Scanned</div></div>
  <div class="stat"><div class="v">{confirmed}</div><div class="l">Confirmed Situations</div></div>
  <div class="stat"><div class="v">{likely}</div><div class="l">Likely Situations</div></div>
  <div class="stat"><div class="v">{total_signals}</div><div class="l">Total Signals Found</div></div>
</div>

{company_sections}

<div class="footer">
  XIMPAX Intelligence Engine — Automated Weekly Report &nbsp;|&nbsp; Powered by Gemini AI<br>
  Questions? Reply to this email.
</div>
</body>
</html>"""


def build_full_report(companies_with_rows: list[tuple]) -> str:
    date_str = datetime.utcnow().strftime("%B %d, %Y")
    confirmed = 0
    likely = 0
    total_signals = 0

    sections = []
    for company, rows in companies_with_rows:
        for row in rows:
            sit = row["situation_status"].lower()
            if "confirmed" in sit: confirmed += 1
            elif "likely" in sit:  likely += 1
            if row["detected_signal"] and "no signals" not in row["detected_signal"].lower():
                total_signals += 1
        sections.append(build_company_section(company, rows))

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
