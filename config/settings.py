"""
Central configuration loaded from environment variables / .env file.
All agents import Settings from here.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Brokers ───────────────────────────────────────────────────────────────
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_paper: bool = True

    oanda_access_token: str
    oanda_account_id: str
    oanda_environment: str = "practice"  # "practice" | "live"

    # ── AI / Claude ───────────────────────────────────────────────────────────
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"

    # ── Sentiment sources ─────────────────────────────────────────────────────
    newsapi_key: str
    twitter_bearer_token: str = ""  # Optional: empty = disabled, no error

    # ── Notifications ─────────────────────────────────────────────────────────
    telegram_bot_token: str
    telegram_chat_id: str

    # ── Risk parameters ───────────────────────────────────────────────────────
    max_risk_per_trade: float = Field(default=0.02, description="2% of portfolio")
    max_daily_drawdown: float = Field(default=0.06, description="6% halt trigger")
    max_concurrent_positions: int = Field(default=3)
    min_signal_confidence: float = Field(default=0.65, description="XGBoost threshold")
    atr_stop_multiplier: float = Field(default=1.5, description="Stop = 1.5×ATR")
    reward_risk_ratio: float = Field(default=2.0, description="TP = 2× stop distance")
    session_blackout_minutes: int = Field(default=5, description="No trades first/last N min")

    # ── Retraining ────────────────────────────────────────────────────────────
    retrain_after_n_trades: int = Field(default=50)
    retrain_min_improvement: float = Field(default=0.05, description="5% Sharpe improvement")

    # ── Database ──────────────────────────────────────────────────────────────
    db_path: str = "data/trading.db"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere
settings = Settings()
