"""
Legacy entry point for DiscordChatBotxOllama.
Maintained for backwards compatibility. Delegates to `main.py` and the `bot` package.
"""

from bot import (
    BotConfig,
    ImageProcessor,
    MemoryManager,
    OllamaService,
    TextSplitter,
    TsundereBot,
    VisionService,
)
from bot.config import DEFAULT_SYSTEM_PROMPT as SYSTEM_PROMPT
from main import main

__all__ = [
    "SYSTEM_PROMPT",
    "BotConfig",
    "ImageProcessor",
    "MemoryManager",
    "OllamaService",
    "TextSplitter",
    "TsundereBot",
    "VisionService",
    "main",
]


if __name__ == "__main__":
    main()