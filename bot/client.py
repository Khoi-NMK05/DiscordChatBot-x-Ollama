import logging

import discord
from discord.ext import commands

from bot.config import BotConfig
from bot.memory import MemoryManager
from bot.services.ollama_service import OllamaService
from bot.services.vision_service import VisionService
from bot.utils.image import ImageProcessor
from bot.utils.text import TextSplitter

logger = logging.getLogger(__name__)


class TsundereBot(commands.Bot):
    """
    Custom Discord Bot client subclass that encapsulates bot lifecycle,
    service injection, and cog extension management.
    """

    def __init__(
        self,
        config: BotConfig,
        memory_manager: MemoryManager | None = None,
        ollama_service: OllamaService | None = None,
        vision_service: VisionService | None = None,
        image_processor: ImageProcessor | None = None,
        text_splitter: TextSplitter | None = None,
        **bot_options,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(
            command_prefix=config.command_prefix,
            intents=intents,
            **bot_options,
        )

        self.config = config
        self.memory_manager = memory_manager or MemoryManager(max_turns=config.max_history_turns)
        self.ollama_service = ollama_service or OllamaService(config=config)
        self.vision_service = vision_service or VisionService(config=config)
        self.image_processor = image_processor or ImageProcessor(max_dimension=config.vision_max_dimension)
        self.text_splitter = text_splitter or TextSplitter(default_limit=1950)

    async def setup_hook(self) -> None:
        """Asynchronous initialization hook called before the bot connects to Discord gateway."""
        logger.info("Loading extensions/cogs...")
        await self.load_extension("bot.cogs.commands")
        await self.load_extension("bot.cogs.chat")
        logger.info("Extensions loaded successfully.")

    async def on_ready(self) -> None:
        """Lifecycle event triggered when the bot is fully connected and ready."""
        if self.user:
            logger.info("Logged in as %s (%s)", self.user.name, self.user.id)
            print(f"Logged in as {self.user.name} ({self.user.id})")

        logger.info("Configured Ollama model: %s", self.config.ollama_model)
        print(f"Connected to Ollama using model: {self.config.ollama_model}")

        if self.config.vision_enabled:
            logger.info(
                "Configured Vision model: %s (max_dim=%d)",
                self.config.vision_model,
                self.config.vision_max_dimension,
            )
            print(f"Connected to Vision using model: {self.config.vision_model}")

        await self.change_presence(activity=discord.Game(name=self.config.presence_text))


    def run_bot(self) -> None:
        """Starts the Discord bot using the configured token."""
        self.run(self.config.discord_token)
