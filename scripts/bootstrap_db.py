"""
First-run script: creates all SQLite tables.
Run once before starting the trading system.

Usage:
    python scripts/bootstrap_db.py
"""
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database import create_all_tables


def main():
    db_path = os.environ.get("DB_PATH", "data/trading.db")
    print(f"Bootstrapping database at: {db_path}")
    create_all_tables(db_path)
    print("All tables created successfully.")


if __name__ == "__main__":
    main()
