"""
Chart generator: creates equity curve PNG for Telegram reports.
"""
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def generate_equity_curve(
    balance_history: list[tuple[datetime, float]],
    output_path: str = "/tmp/equity_curve.png",
) -> Optional[str]:
    """
    Generate an equity curve chart from a list of (timestamp, balance) tuples.

    Returns:
        Path to the saved PNG, or None on error.
    """
    if len(balance_history) < 2:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        timestamps = [t for t, _ in balance_history]
        balances = [b for _, b in balance_history]
        initial = balances[0]

        pnl_pct = [(b - initial) / initial * 100 for b in balances]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.patch.set_facecolor("#1a1a2e")

        # Balance curve
        ax1.plot(timestamps, balances, color="#00d4ff", linewidth=1.5)
        ax1.fill_between(timestamps, balances, alpha=0.1, color="#00d4ff")
        ax1.set_facecolor("#16213e")
        ax1.set_ylabel("Balance ($)", color="white", fontsize=9)
        ax1.tick_params(colors="white", labelsize=8)
        ax1.spines["bottom"].set_color("#444")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_color("#444")
        ax1.grid(alpha=0.2, color="gray")

        # P&L % curve
        colors = ["#00ff88" if p >= 0 else "#ff4444" for p in pnl_pct]
        ax2.bar(timestamps, pnl_pct, color=colors, alpha=0.7, width=0.01)
        ax2.axhline(y=0, color="white", linewidth=0.5, alpha=0.5)
        ax2.set_facecolor("#16213e")
        ax2.set_ylabel("P&L (%)", color="white", fontsize=9)
        ax2.tick_params(colors="white", labelsize=8)
        ax2.spines["bottom"].set_color("#444")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_color("#444")
        ax2.grid(alpha=0.2, color="gray")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

        total_pnl_pct = pnl_pct[-1] if pnl_pct else 0.0
        color = "#00ff88" if total_pnl_pct >= 0 else "#ff4444"
        fig.suptitle(
            f"Trading Bot — Equity Curve  ({total_pnl_pct:+.2f}%)",
            color=color, fontsize=11, fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path

    except Exception as exc:
        logger.error("Chart generation failed: %s", exc)
        return None
