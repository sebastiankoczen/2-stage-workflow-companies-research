"""
XIMPAX Intelligence Engine — Orchestrator
Runs Stage 1 → Stage 2 in sequence.
Called by the GitHub Actions weekly workflow.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    # ── Stage 1 ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("XIMPAX INTELLIGENCE ENGINE — STAGE 1 START")
    log.info("=" * 60)

    try:
        import stage1_run
        html_path, stage2_input = stage1_run.main()
        log.info(f"Stage 1 complete → {html_path}")
    except Exception as e:
        log.error(f"Stage 1 FAILED: {e}")
        sys.exit(1)

    # ── Stage 2 ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("XIMPAX INTELLIGENCE ENGINE — STAGE 2 START")
    log.info("=" * 60)

    try:
        import stage2_run
        report_path = stage2_run.main()
        log.info(f"Stage 2 complete → {report_path}")
    except Exception as e:
        log.error(f"Stage 2 FAILED: {e}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("✅ XIMPAX INTELLIGENCE ENGINE COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
