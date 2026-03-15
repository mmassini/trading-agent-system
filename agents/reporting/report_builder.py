"""
Report builder: composes the 12-hour trading summary for Telegram.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from storage.database import Database

logger = logging.getLogger(__name__)


def build_report(db: Database) -> str:
    """
    Compose a Telegram-ready HTML report of the last 12 hours.
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=12)

    # ── Balance ───────────────────────────────────────────────────────────────
    current_balance = db.get_latest_balance("combined") or 0.0
    balance_12h_ago = _get_balance_at(db, since) or current_balance
    pnl_12h = current_balance - balance_12h_ago
    pnl_12h_pct = (pnl_12h / balance_12h_ago * 100) if balance_12h_ago > 0 else 0.0

    # First snapshot ever (for cumulative P&L)
    initial_balance = _get_first_balance(db) or current_balance
    cum_pnl = current_balance - initial_balance
    cum_pct = (cum_pnl / initial_balance * 100) if initial_balance > 0 else 0.0

    # ── Recent trades ─────────────────────────────────────────────────────────
    recent_trades = _get_recent_trades(db, since)
    total_recent = len(recent_trades)
    wins = [t for t in recent_trades if (t.pnl or 0) > 0]
    losses = [t for t in recent_trades if (t.pnl or 0) <= 0]
    win_rate = len(wins) / total_recent * 100 if total_recent > 0 else 0.0

    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    best = max((t.pnl for t in recent_trades), default=0.0)
    worst = min((t.pnl for t in recent_trades), default=0.0)
    best_symbol = next((t.symbol for t in recent_trades if t.pnl == best), "—")
    worst_symbol = next((t.symbol for t in recent_trades if t.pnl == worst), "—")

    # ── Open positions ────────────────────────────────────────────────────────
    open_trades = db.get_open_trades()

    # ── Model info ────────────────────────────────────────────────────────────
    model_version = _get_champion_version()

    # ── Compose HTML ─────────────────────────────────────────────────────────
    pnl_color = "🟢" if pnl_12h >= 0 else "🔴"
    cum_color = "🟢" if cum_pnl >= 0 else "🔴"

    lines = [
        f"<b>Trading Report — {now.strftime('%Y-%m-%d %H:%M')} UTC</b>",
        "",
        f"<b>Portfolio Balance:</b> ${current_balance:,.2f}",
        f"<b>12h P&amp;L:</b> {pnl_color} ${pnl_12h:+,.2f} ({pnl_12h_pct:+.2f}%)",
        f"<b>Cumulative P&amp;L:</b> {cum_color} ${cum_pnl:+,.2f} ({cum_pct:+.2f}%)",
        "",
        f"<b>Trades (last 12h):</b> {total_recent}",
    ]

    if total_recent > 0:
        lines += [
            f"  Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)",
            f"  Avg Win: +${avg_win:,.2f}  |  Avg Loss: -${abs(avg_loss):,.2f}",
            f"  Best: {best_symbol} +${best:,.2f}",
            f"  Worst: {worst_symbol} -${abs(worst):,.2f}",
        ]

    lines += [""]

    if open_trades:
        lines.append(f"<b>Open Positions ({len(open_trades)}):</b>")
        for t in open_trades[:5]:  # Max 5 shown
            direction_arrow = "▲" if t.direction == "LONG" else "▼"
            lines.append(
                f"  {direction_arrow} {t.symbol}  {int(t.quantity)}  "
                f"Entry: ${t.entry_price:.4f}  "
                f"SL: ${t.stop_loss:.4f}  TP: ${t.take_profit:.4f}"
            )
    else:
        lines.append("<b>Open Positions:</b> None")

    lines += [
        "",
        f"<i>Model: {model_version}</i>",
    ]

    return "\n".join(lines)


def _get_balance_at(db: Database, timestamp) -> Optional[float]:
    from sqlalchemy import func
    from storage.schema import BalanceSnapshot
    with db.session() as s:
        snap = (
            s.query(BalanceSnapshot)
            .filter(
                BalanceSnapshot.source == "combined",
                BalanceSnapshot.snapshot_at <= timestamp,
            )
            .order_by(BalanceSnapshot.snapshot_at.desc())
            .first()
        )
        return snap.balance if snap else None


def _get_first_balance(db: Database) -> Optional[float]:
    from storage.schema import BalanceSnapshot
    with db.session() as s:
        snap = (
            s.query(BalanceSnapshot)
            .filter(BalanceSnapshot.source == "combined")
            .order_by(BalanceSnapshot.snapshot_at.asc())
            .first()
        )
        return snap.balance if snap else None


def _get_recent_trades(db: Database, since):
    from storage.schema import Trade
    with db.session() as s:
        trades = (
            s.query(Trade)
            .filter(Trade.status == "closed", Trade.exit_at >= since)
            .order_by(Trade.exit_at.desc())
            .all()
        )
        s.expunge_all()
        return trades


def _get_champion_version() -> str:
    import json
    from pathlib import Path
    path = Path("models/champion/metrics.json")
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("stock_version", "unknown")
        except Exception:
            pass
    return "not trained"
