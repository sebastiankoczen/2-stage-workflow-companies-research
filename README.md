# XIMPAX Intelligence Engine

Automated 2-stage weekly supply chain market intelligence system.  
Runs every Monday, delivers a deep-scan report straight to your inbox.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  GitHub Actions (weekly cron — every Monday 06:00 UTC)      │
│                                                             │
│  STAGE 1 — Discovery                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Prompt 1 + Company Profile + ICP + Instructions    │   │
│  │  → Run 10× through Gemini API                       │   │
│  │  → Aggregate + deduplicate all results              │   │
│  │  → Generate HTML chart (sorted by Priority Score)   │   │
│  │  → Export Top 10 Tier 1 companies → stage2_input    │   │
│  └─────────────────────────────────────────────────────┘   │
│              ↓                                              │
│  STAGE 2 — Deep Scan                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Top 10 companies × Prompt 2 → Gemini API           │   │
│  │  → Signal Strength Matrix per company               │   │
│  │  → Generate polished HTML email report              │   │
│  │  → Send via Gmail SMTP → your work email            │   │
│  └─────────────────────────────────────────────────────┘   │
│              ↓                                              │
│  (Optional) Power Automate distributes to your list         │
└─────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
ximpax-intelligence/
├── .github/
│   └── workflows/
│       └── weekly_intelligence.yml   ← GitHub Actions schedule
├── config/
│   ├── company_profile.txt           ← XIMPAX firm profile
│   ├── icp_blueprint.txt             ← Ideal Customer Profile
│   └── instructions.txt             ← Situation detection framework
├── prompts/
│   ├── prompt_1.txt                  ← Stage 1 discovery prompt
│   └── prompt_2.txt                  ← Stage 2 deep scan prompt
├── scripts/
│   ├── orchestrate.py                ← Runs both stages in sequence
│   ├── stage1_run.py                 ← Stage 1 logic
│   └── stage2_run.py                 ← Stage 2 logic + email
├── output/                           ← Generated reports (git-tracked)
│   ├── stage1_chart_YYYYMMDD.html
│   ├── stage2_report_YYYYMMDD.html
│   └── stage2_input.json
├── requirements.txt
└── README.md
```

---

## Setup Guide (one-time)

### Step 1 — Fork / Clone this Repo

```bash
git clone https://github.com/YOUR_ORG/ximpax-intelligence.git
cd ximpax-intelligence
```

### Step 2 — Get a Gemini API Key

1. Go to **https://aistudio.google.com/app/apikey**
2. Click **Create API Key**
3. Copy the key — you'll need it in Step 4

> **Free tier limits:** Gemini 1.5 Pro allows 2 RPM free. The scripts include  
> automatic sleep pauses between runs to stay within limits.  
> For faster execution, upgrade to a paid Gemini tier.

### Step 3 — Set Up Gmail for Sending

You need a Gmail account dedicated to sending (e.g. `ximpax.reports@gmail.com`):

1. Enable 2-Step Verification on that Gmail account  
2. Go to **Google Account → Security → App Passwords**
3. Generate an App Password (select "Mail" + "Other")
4. Copy the 16-character password

### Step 4 — Add GitHub Secrets

In your GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Value |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key from Step 2 |
| `GMAIL_ADDRESS` | The Gmail address you set up in Step 3 |
| `GMAIL_APP_PASSWORD` | The 16-char App Password from Step 3 |
| `RECIPIENT_EMAIL` | Your **work email** where you want to receive reports |

### Step 5 — Enable GitHub Actions

Go to your repo → **Actions** tab → Click **"I understand my workflows, go ahead and enable them"**

That's it. The workflow runs every **Monday at 06:00 UTC**.

---

## Running Manually

You can trigger a run anytime from the GitHub UI:

1. Go to **Actions → XIMPAX Weekly Intelligence Run**
2. Click **"Run workflow"** → **"Run workflow"**

Or locally (for testing):

```bash
# Install deps
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY=your_key_here
export GMAIL_ADDRESS=your_gmail@gmail.com
export GMAIL_APP_PASSWORD=your_app_password
export RECIPIENT_EMAIL=your_work@email.com

# Run both stages
cd scripts
python orchestrate.py

# Or run stages individually:
python stage1_run.py   # Stage 1 only
python stage2_run.py   # Stage 2 only (requires stage1 output)
```

---

## Output Files

Every run produces 4 files in `/output`:

| File | Description |
|---|---|
| `stage1_chart_YYYYMMDD.html` | Full discovery chart, all companies, sortable |
| `stage1_raw_YYYYMMDD.json` | Raw AI responses from all 10 runs (audit trail) |
| `stage2_input.json` | Top 10 companies passed to Stage 2 |
| `stage2_report_YYYYMMDD.html` | Deep scan report — this is what gets emailed |

All outputs are committed back to the repo automatically as an audit trail.

---

## Power Automate Distribution (Final Step)

To distribute the report to your team list after it arrives in your work inbox:

1. Open **Power Automate** → **Create** → **Automated cloud flow**
2. Trigger: **"When a new email arrives (V3)"**
   - Folder: Inbox
   - Filter Subject contains: `XIMPAX Weekly Intelligence`
3. Action: **"Forward an email (V2)"**
   - To: your distribution list or individual addresses
   - Or use a **"Send an email (V2)"** action to customize the body

This way you control distribution without giving anyone else access to the GitHub repo.

---

## Adjusting Campaign Filters

Edit these values at the top of `scripts/stage1_run.py`:

```python
NUM_RUNS = 10               # Number of Gemini runs per week
TOP_N_FOR_STAGE2 = 10       # How many top companies go to Stage 2
```

To change the target region, industry, or revenue range, edit the  
`{default = ...}` placeholders in `prompts/prompt_1.txt`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `GEMINI_API_KEY` error | Check the secret is set correctly in GitHub Settings |
| Email not received | Check Gmail App Password; verify 2FA is enabled |
| 0 rows parsed from AI | Gemini output format changed — check `stage1_raw_*.json` |
| Rate limit errors | Increase `SLEEP_SECONDS` in `stage1_run.py` |
| Workflow not running | Check Actions are enabled; verify cron syntax |

---

## Customisation

- **Switch from Gemini to another model**: Replace `call_gemini()` in both stage scripts with your preferred API client (OpenAI, Anthropic, Perplexity)
- **Change schedule**: Edit the `cron` expression in `.github/workflows/weekly_intelligence.yml`
- **Add more runs**: Change `NUM_RUNS = 10` to any number
- **Different Top N**: Change `TOP_N_FOR_STAGE2 = 10`
