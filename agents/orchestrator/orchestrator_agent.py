"""
Orchestrator Agent: the brain of the trading system.

Coordinates all subagents in the decision pipeline:
  1. Check session gates (market hours, drawdown halt)
  2. Collect ML signals for all active symbols (parallel)
  3. Collect sentiment scores for all active symbols (parallel)
  4. Evaluate each signal through the Risk Agent
  5. Submit approved orders via the Execution Agent
  6. Trigger Post-Mortem after each trade close (via OrderTracker events)
  7. Record heartbeats to SQLite for health monitoring

Decision cycle runs every 30 seconds during active market hours.
"""
import asyncio
import logging
import threading
from datetime import datetime, timezone

import yaml

from agents.data_ingest.alpaca_stream import AlpacaStreamHandler
from agents.data_ingest.bar_aggregator import BarAggregator
from agents.data_ingest.data_buffer import DataBuffer
from agents.data_ingest.oanda_stream import OandaStreamHandler
from agents.execution.alpaca_executor import AlpacaExecutor
from agents.execution.execution_agent import ExecutionAgent
from agents.execution.oanda_executor import OandaExecutor
from agents.execution.order_tracker import OrderTracker
from agents.ml_analysis.ml_agent import analyze_symbol
from agents.ml_analysis.model_registry import ModelRegistry
from agents.orchestrator.session_manager import SessionManager
from agents.postmortem.postmortem_agent import PostMortemAgent
from agents.risk.risk_agent import RiskAgent
from agents.reporting.telegram_client import TelegramClient
from agents.sentiment.sentiment_agent import SentimentAgent
from config.settings import Settings
from storage.database import Database

logger = logging.getLogger(__name__)

DECISION_INTERVAL_SECONDS = 30
HEARTBEAT_INTERVAL_SECONDS = 60


class OrchestratorAgent:
    """
    Main trading agent. Runs the decision loop and manages all subagents.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._running = False

        # ── Core infrastructure ───────────────────────────────────────────
        self._db = Database(settings.db_path)
        self._buffer = DataBuffer(maxlen=200)

        # ── Instruments ───────────────────────────────────────────────────
        with open("config/instruments.yaml") as f:
            instruments = yaml.safe_load(f)
        self._stocks = instruments.get("stocks", [])
        self._forex = instruments.get("forex", [])
        self._all_symbols = self._stocks + self._forex

        # ── Data ingest ───────────────────────────────────────────────────
        self._aggregator = BarAggregator(
            buffer=self._buffer,
            on_bar_complete=self._persist_bar,
        )

        self._alpaca_stream = AlpacaStreamHandler(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            symbols=self._stocks,
            aggregator=self._aggregator,
            paper=settings.alpaca_paper,
        )

        # OANDA only initialized if credentials are provided and forex list is non-empty
        self._oanda_enabled = bool(settings.oanda_access_token and self._forex)
        self._oanda_stream = OandaStreamHandler(
            access_token=settings.oanda_access_token,
            account_id=settings.oanda_account_id,
            instruments=self._forex if self._oanda_enabled else [],
            aggregator=self._aggregator,
            environment=settings.oanda_environment,
        ) if self._oanda_enabled else None

        # ── Broker executors ──────────────────────────────────────────────
        self._alpaca_exec = AlpacaExecutor(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.alpaca_paper,
        )
        self._oanda_exec = OandaExecutor(
            access_token=settings.oanda_access_token,
            account_id=settings.oanda_account_id,
            environment=settings.oanda_environment,
        ) if self._oanda_enabled else None

        # ── Agents ────────────────────────────────────────────────────────
        self._session = SessionManager(
            blackout_minutes=settings.session_blackout_minutes
        )
        self._model_registry = ModelRegistry()
        self._sentiment = SentimentAgent(
            newsapi_key=settings.newsapi_key,
            twitter_bearer_token=settings.twitter_bearer_token,
        )
        self._risk = RiskAgent(
            db=self._db,
            max_risk_pct=settings.max_risk_per_trade,
            atr_multiplier=settings.atr_stop_multiplier,
            reward_risk_ratio=settings.reward_risk_ratio,
            min_confidence=settings.min_signal_confidence,
            max_daily_drawdown=settings.max_daily_drawdown,
            max_positions=settings.max_concurrent_positions,
        )
        self._execution = ExecutionAgent(
            db=self._db,
            alpaca=self._alpaca_exec,
            oanda=self._oanda_exec,
        )
        self._postmortem = PostMortemAgent(
            db=self._db,
            settings=settings,
        )
        self._telegram = TelegramClient(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        self._order_tracker = OrderTracker(
            db=self._db,
            alpaca_executor=self._alpaca_exec,
            oanda_executor=self._oanda_exec,
            poll_interval=10.0,
        )

    def run(self):
        """
        Entry point: starts all background threads and then runs the async decision loop.
        """
        logger.info("Orchestrator starting...")
        self._running = True

        # Start data streams in background threads
        threading.Thread(
            target=self._alpaca_stream.start, daemon=True, name="AlpacaStream"
        ).start()
        if self._oanda_stream:
            threading.Thread(
                target=self._oanda_stream.start, daemon=True, name="OandaStream"
            ).start()

        # Start order tracker
        self._order_tracker.start()

        # Run async decision loop
        asyncio.run(self._main_loop())

    async def _main_loop(self):
        """Async main loop: runs the decision cycle every 30 seconds."""
        heartbeat_counter = 0
        market_was_active = False
        session_started_notified = False

        while self._running:
            try:
                market_active = self._session.is_stock_session_active()
            except Exception as exc:
                logger.error("Session check error: %s", exc, exc_info=True)
                await asyncio.sleep(DECISION_INTERVAL_SECONDS)
                continue

            # Notify Telegram when market session begins
            if market_active and not session_started_notified:
                session_started_notified = True
                await self._telegram.send_message(
                    "Trading session started\n"
                    f"Symbols: {', '.join(self._stocks)}\n"
                    "Waiting for data (first 50 bars)..."
                )

            # Detect market close transition → EOD cleanup then exit
            if market_was_active and not market_active:
                logger.info("Market closed — running EOD position close.")
                await self._eod_close()
                logger.info("EOD complete. Shutting down.")
                self._running = False
                break

            market_was_active = market_active

            try:
                await self._decision_cycle()
            except Exception as exc:
                logger.error("Decision cycle error: %s", exc, exc_info=True)
                await self._telegram.send_message(f"⚠️ Decision cycle error: {exc}")

            # Heartbeat every N cycles
            heartbeat_counter += 1
            if heartbeat_counter * DECISION_INTERVAL_SECONDS >= HEARTBEAT_INTERVAL_SECONDS:
                try:
                    self._db.record_heartbeat("orchestrator", "ok")
                except Exception:
                    pass
                heartbeat_counter = 0

            try:
                await asyncio.sleep(DECISION_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                logger.info("Main loop cancelled — shutting down.")
                self._running = False
                break

    async def _decision_cycle(self):
        """
        One complete decision cycle:
          1. Check model for updates
          2. Collect account balance
          3. For each symbol, run ML + Sentiment in parallel
          4. Risk-check each signal
          5. Execute approved trades
        """
        # Hot-reload champion model if updated
        self._model_registry.check_and_reload()

        # Get current account balance for position sizing
        account_balance = self._get_combined_balance()
        if account_balance <= 0:
            logger.warning("Could not determine account balance — skipping cycle")
            return

        # Collect active symbols (only those with enough bar data)
        active_symbols = [
            s for s in self._all_symbols
            if (
                self._session.is_tradeable_for_symbol(s)
                and self._buffer.has_enough_data(s, "1m", min_bars=50)
            )
        ]

        if not active_symbols:
            return

        logger.debug("Decision cycle: %d active symbols, balance=$%.2f",
                     len(active_symbols), account_balance)

        # Run ML and Sentiment in parallel for all symbols
        tasks = [
            self._analyze_symbol_async(s, account_balance)
            for s in active_symbols
        ]
        await asyncio.gather(*tasks)

    async def _analyze_symbol_async(self, symbol: str, account_balance: float):
        """
        Analyze a single symbol: ML signal + sentiment → risk check → execute.
        Runs in asyncio task (not truly parallel IO, but non-blocking for CPU).
        """
        # Run blocking calls in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()

        # Get sentiment score (cached, fast after first call)
        sentiment = await loop.run_in_executor(
            None, self._sentiment.get_score, symbol
        )

        # Get ML signal
        signal = await loop.run_in_executor(
            None,
            analyze_symbol,
            symbol,
            self._buffer,
            self._model_registry,
            sentiment.score,
        )

        if not signal.is_actionable(self._settings.min_signal_confidence):
            return

        # Get current price for order sizing
        last_bar = self._buffer.get_last(symbol, "1m")
        if last_bar is None:
            return
        current_price = last_bar.close

        # Risk evaluation
        decision = self._risk.evaluate(signal, current_price, account_balance)

        if not decision.approved:
            logger.debug("Risk rejected %s: %s", symbol, decision.reason)
            return

        # Execute the trade
        trade_id = self._execution.execute(
            spec=decision.order_spec,
            signal_confidence=signal.confidence,
            sentiment_score=sentiment.score,
            model_version=signal.model_version,
        )

        if trade_id:
            logger.info(
                "Trade opened: %s | confidence=%.2f | sentiment=%.3f",
                symbol, signal.confidence, sentiment.score,
            )
            await self._telegram.send_message(
                f"Trade opened\n"
                f"{decision.order_spec.direction} {symbol} x{decision.order_spec.quantity}\n"
                f"Entry: ${current_price:.2f}  SL: ${decision.order_spec.stop_loss:.2f}  TP: ${decision.order_spec.take_profit:.2f}\n"
                f"Confidence: {signal.confidence:.0%}  Sentiment: {sentiment.score:+.2f}"
            )

            # Trigger post-mortem check (non-blocking, async)
            asyncio.create_task(self._check_postmortem_async())

    async def _check_postmortem_async(self):
        """Trigger retraining if threshold reached."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._postmortem.check_retrain_threshold)

    async def _eod_close(self):
        """Close all open positions at end of day."""
        try:
            n = self._alpaca_exec.close_all_positions()
            if n:
                logger.info("EOD: %d Alpaca position(s) closed.", n)
            else:
                logger.info("EOD: no open positions to close.")
            balance = self._alpaca_exec.get_account_balance()
            await self._telegram.send_message(
                f"Session ended\n"
                f"Positions closed: {n}\n"
                f"Account equity: ${balance:,.2f}"
            )
        except Exception as exc:
            logger.error("EOD close error: %s", exc)

    def _get_combined_balance(self) -> float:
        balance = self._db.get_latest_balance("combined")
        if balance:
            return balance
        # Fallback: query brokers directly
        try:
            alpaca_bal = self._alpaca_exec.get_account_balance()
            oanda_bal = self._oanda_exec.get_account_balance() if self._oanda_exec else 0.0
            return alpaca_bal + oanda_bal
        except Exception:
            return 0.0

    def _persist_bar(self, bar):
        """Persist a completed bar to SQLite (called by BarAggregator)."""
        from storage.schema import Bar
        db_bar = Bar(
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            timestamp=bar.timestamp,
            source="live",
        )
        self._db.save_trade  # Reuse session context
        with self._db.session() as s:
            s.add(db_bar)

    def stop(self):
        self._running = False
        self._alpaca_stream.stop()
        if self._oanda_stream:
            self._oanda_stream.stop()
        self._order_tracker.stop()
        logger.info("Orchestrator stopped.")
