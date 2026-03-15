"""
Entry point for the Trading Agent System.
Bootstraps the database, wires up all components, and starts the orchestrator.
"""
import logging
import os
import signal
import sys

# Configure logging before any imports
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("main")


def main():
    logger.info("Trading Agent System starting...")

    # Load settings (reads .env or environment variables)
    from config.settings import settings

    # Bootstrap database
    from storage.database import Database
    db = Database(settings.db_path)
    logger.info("Database ready: %s", settings.db_path)

    # Start the orchestrator
    from agents.orchestrator.orchestrator_agent import OrchestratorAgent
    orchestrator = OrchestratorAgent(settings)

    # Graceful shutdown on SIGTERM/SIGINT
    def shutdown(signum, frame):
        logger.info("Shutdown signal received — stopping gracefully...")
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logger.info("Orchestrator starting. Press Ctrl+C to stop.")
    orchestrator.run()


if __name__ == "__main__":
    main()
