import asyncio
import logging

import discord
from discord.ext import commands

from bot.memory import MemoryManager
from bot.services.ollama_service import OllamaService
from bot.services.vision_service import VisionService
from bot.utils.image import ImageProcessor
from bot.utils.text import TextSplitter

logger = logging.getLogger(__name__)


class ChatCog(commands.Cog, name="Chat"):
    """Handles chat events, mention detection, vision perception, and AI interactions."""

    def __init__(
        self,
        bot: commands.Bot,
        ollama_service: OllamaService,
        memory_manager: MemoryManager,
        vision_service: VisionService | None = None,
        image_processor: ImageProcessor | None = None,
        text_splitter: TextSplitter | None = None,
    ) -> None:
        self.bot = bot
        self.ollama_service = ollama_service
        self.memory_manager = memory_manager
        self.vision_service = vision_service
        self.image_processor = image_processor or ImageProcessor()
        self.text_splitter = text_splitter or TextSplitter(default_limit=1950)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        # Ignore messages sent by bots
        if message.author.bot:
            return

        # Check if the message is a valid command so we don't trigger chat
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            return

        # Trigger only when pinged directly (@bot)
        if self.bot.user and self.bot.user in message.mentions:
            await self._handle_bot_mention(message)

    async def _extract_image_description(
        self,
        message: discord.Message,
        user_query: str | None = None,
    ) -> str | None:
        """
        Extracts, downscales, and describes the first supported image attachment
        using the decoupled vision service. Discards raw image bytes immediately
        to ensure ephemeral memory handling. Incorporates user_query for context-aware
        analysis and OCR transcription.
        """
        if not self.vision_service or not self.vision_service.config.vision_enabled:
            return None

        # Filter for supported image attachments
        image_attachments = [
            att
            for att in message.attachments
            if (att.content_type and self.image_processor.is_supported_mime_type(att.content_type))
            or self.image_processor.is_supported_filename(att.filename)
        ]

        if not image_attachments:
            return None

        attachment = image_attachments[0]
        try:
            logger.info("Downloading and processing image attachment '%s'...", attachment.filename)
            raw_bytes = await attachment.read()
            # Downscale asynchronously via worker thread
            processed_bytes = await self.image_processor.process_image(raw_bytes)
            # Ephemeral: release raw download bytes immediately
            del raw_bytes

            # Describe image using decoupled VLM with user query context
            description = await self.vision_service.describe_image(
                processed_bytes,
                user_query=user_query,
            )
            # Ephemeral: release processed image bytes
            del processed_bytes

            return description if description else None

        except Exception as err:
            logger.error("Failed to analyze image attachment '%s': %s", attachment.filename, err)
            return None

    async def _handle_bot_mention(self, message: discord.Message) -> None:
        assert self.bot.user is not None

        # Clean prompt by stripping bot mention
        user_prompt = message.clean_content.replace(f"@{self.bot.user.display_name}", "").strip()

        # Check for image attachments if user provided no text or provided both
        has_attachments = bool(message.attachments)
        if not user_prompt and not has_attachments:
            await message.reply("You pinged me, but didn't say anything! What's on your mind?")
            return

        guild_id = message.guild.id if message.guild else None
        channel_id = message.channel.id

        # Show typing indicator while generating response
        async with message.channel.typing():
            # If an image is attached, describe it via decoupled vision pipeline with query context
            image_description = await self._extract_image_description(message, user_query=user_prompt)

            if not user_prompt and image_description:
                user_prompt = "What do you think of this?"


            # Format user prompt and convert image description into pure text
            if image_description:
                formatted_prompt = (
                    f"{message.author.display_name}: [Attached Image: {image_description}]\n{user_prompt}"
                )
            else:
                formatted_prompt = f"{message.author.display_name}: {user_prompt}"

            history = self.memory_manager.get_history(guild_id, channel_id)

            try:
                llm_response = await self.ollama_service.generate_response(
                    history=history,
                    formatted_user_prompt=formatted_prompt,
                )

                # Add reaction emoji if the model chose one
                if llm_response.reaction:
                    try:
                        await message.add_reaction(llm_response.reaction)
                    except discord.HTTPException:
                        logger.warning("Failed to add reaction '%s'", llm_response.reaction)

                # Update channel memory (Text only: zero visual tokens in KV-cache)
                self.memory_manager.add_turn(
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_content=formatted_prompt,
                    assistant_content=llm_response.content,
                )

                # Split message into chunks (either separate continuous chat messages or single-block chunking)
                config = self.ollama_service.config
                if config.separate_messages:
                    reply_chunks = self.text_splitter.split_chat_messages(llm_response.content)
                else:
                    reply_chunks = self.text_splitter.split(llm_response.content)

                if not reply_chunks:
                    reply_chunks = [llm_response.content] if llm_response.content.strip() else []

                for index, chunk in enumerate(reply_chunks):
                    if index == 0:
                        await message.reply(chunk, mention_author=False)
                    else:
                        if config.separate_messages and config.message_delay > 0:
                            try:
                                async with message.channel.typing():
                                    await asyncio.sleep(config.message_delay)
                            except Exception:
                                await asyncio.sleep(config.message_delay)
                        await message.channel.send(chunk)

            except Exception:
                logger.exception("Error generating response from Ollama")
                await message.reply(
                    "⚠️ I ran into an error generating a response. Make sure Ollama is running locally."
                )


async def setup(bot: commands.Bot) -> None:
    # Service dependencies are injected via the bot instance attributes
    ollama_service = getattr(bot, "ollama_service", None)
    memory_manager = getattr(bot, "memory_manager", None)
    vision_service = getattr(bot, "vision_service", None)
    image_processor = getattr(bot, "image_processor", None)
    text_splitter = getattr(bot, "text_splitter", None)

    if not ollama_service or not memory_manager:
        raise RuntimeError("Bot is missing required services (ollama_service, memory_manager).")

    await bot.add_cog(
        ChatCog(
            bot=bot,
            ollama_service=ollama_service,
            memory_manager=memory_manager,
            vision_service=vision_service,
            image_processor=image_processor,
            text_splitter=text_splitter,
        )
    )

