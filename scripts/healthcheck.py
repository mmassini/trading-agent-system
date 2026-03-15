"""
Healthcheck script: verifies the orchestrator sent a heartbeat recently.
Used by the Docker HEALTHCHECK and GitHub Actions healthcheck workflow.

Usage:
    python scripts/healthcheck.py [--max-age-seconds=120]

Exit codes:
    0 = healthy
    1 = unhealthy (stale or missing heartbeat)
"""
import sys
import os
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-age-seconds", type=int, default=120)
    args = parser.parse_args()

    from storage.database import Database
    db = Database(os.environ.get("DB_PATH", "data/trading.db"))

    hb = db.get_last_heartbeat("orchestrator")
    if hb is None:
        print("UNHEALTHY: No heartbeat found for orchestrator")
        sys.exit(1)

    age = (datetime.utcnow() - hb.recorded_at).total_seconds()
    if age > args.max_age_seconds:
        print(f"UNHEALTHY: Last heartbeat was {int(age)}s ago (max {args.max_age_seconds}s)")
        sys.exit(1)

    print(f"HEALTHY: Last heartbeat {int(age)}s ago — {hb.status}")
    sys.exit(0)


if __name__ == "__main__":
    main()
