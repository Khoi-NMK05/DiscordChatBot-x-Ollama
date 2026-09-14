import logging
import sys

from bot import BotConfig, TsundereBot


def setup_logging() -> None:
    """Configures application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """Main application entrypoint."""
    setup_logging()
    logger = logging.getLogger("main")

    try:
        config = BotConfig.from_env()
    except ValueError as e:
        logger.error("Configuration Error: %s", e)
        sys.exit(1)

    logger.info("Initializing TsundereBot...")
    bot = TsundereBot(config=config)

    logger.info("Starting bot...")
    bot.run_bot()


if __name__ == "__main__":
    main()
