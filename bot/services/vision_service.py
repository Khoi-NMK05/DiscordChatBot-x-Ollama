from __future__ import annotations

import base64
import logging
import re
from typing import Any

import ollama

from bot.config import BotConfig

logger = logging.getLogger(__name__)


class VisionServiceError(Exception):
    """Raised when the vision service encounters an inference or connection error."""


class VisionService:
    """
    Decoupled computer vision service interfacing with a dedicated vision model
    (e.g., moondream, qwen2.5-vl) via Ollama.

    Optimized for ephemeral image handling: processes images on-the-fly and returns
    concise textual summaries without storing raw image binary in state or chat history.
    """

    def __init__(self, config: BotConfig, client: ollama.AsyncClient | None = None) -> None:
        self.config = config
        if client is not None:
            self.client = client
        elif config.ollama_host:
            self.client = ollama.AsyncClient(host=config.ollama_host)
        else:
            self.client = ollama.AsyncClient()

    @staticmethod
    def clean_think_tags(text: str) -> str:
        """Strips reasoning blocks from models that emit <think> tags."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    async def describe_image(
        self,
        image_bytes: bytes,
        user_query: str | None = None,
        custom_prompt: str | None = None,
    ) -> str:
        """
        Submits an ephemeral image to the vision model and retrieves an accurate,
        context-aware description or OCR transcription of its visual contents.
        """
        if not self.config.vision_enabled:
            return ""

        if custom_prompt:
            prompt = custom_prompt
        elif user_query and user_query.strip():
            prompt = (
                f'The user sent this image with the message: "{user_query.strip()}"\n'
                "Analyze the image carefully to address their query. "
                "If the image contains text, homework, worksheets, math formulas, code, or documents, "
                "accurately transcribe and extract the relevant text and details. "
                "Describe any relevant visual details accurately."
            )
        else:
            prompt = self.config.vision_prompt

        # Ephemeral base64 encoding (local scope only)
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        messages_payload = [
            {
                "role": "user",
                "content": prompt,
                "images": [encoded_image],
            }
        ]

        num_predict = getattr(self.config, "vision_num_predict", 1024)

        try:
            logger.info("Calling vision model '%s' (predict=%d)...", self.config.vision_model, num_predict)
            response: dict[str, Any] = await self.client.chat(
                model=self.config.vision_model,
                messages=messages_payload,
                think=False,
                options={
                    "temperature": 0.2,
                    "num_predict": num_predict,
                    "num_ctx": 4096,
                },
            )

            raw_description = response.get("message", {}).get("content", "").strip()
            cleaned_description = self.clean_think_tags(raw_description)
            logger.debug("Vision description: %s", cleaned_description)
            return cleaned_description

        except Exception as err:
            logger.error("Vision service failed to describe image: %s", err)
            raise VisionServiceError(f"Failed to analyze image with vision model: {err}") from err

