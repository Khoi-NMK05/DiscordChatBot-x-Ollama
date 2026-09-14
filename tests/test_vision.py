from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

from bot.config import BotConfig
from bot.memory import MemoryManager
from bot.services.vision_service import VisionService, VisionServiceError
from bot.utils.image import ImageProcessingError, ImageProcessor


class TestImageProcessor(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.processor = ImageProcessor(max_dimension=1024, quality=85)


    def _create_synthetic_image(
        self,
        width: int,
        height: int,
        mode: str = "RGB",
        format_name: str = "JPEG",
    ) -> bytes:
        img = Image.new(mode, (width, height), color=(120, 150, 200) if "RGB" in mode else 128)
        buf = io.BytesIO()
        img.save(buf, format=format_name)
        return buf.getvalue()

    def test_supported_mime_and_extension(self) -> None:
        self.assertTrue(self.processor.is_supported_mime_type("image/jpeg"))
        self.assertTrue(self.processor.is_supported_mime_type("image/png; charset=utf-8"))
        self.assertTrue(self.processor.is_supported_mime_type("image/webp"))
        self.assertTrue(self.processor.is_supported_mime_type("image/gif"))
        self.assertFalse(self.processor.is_supported_mime_type("application/pdf"))
        self.assertFalse(self.processor.is_supported_mime_type(None))

        self.assertTrue(self.processor.is_supported_filename("photo.JPG"))
        self.assertTrue(self.processor.is_supported_filename("art.PNG"))
        self.assertTrue(self.processor.is_supported_filename("anim.gif"))
        self.assertFalse(self.processor.is_supported_filename("document.pdf"))

    def test_downscale_large_image(self) -> None:
        raw_bytes = self._create_synthetic_image(2000, 1000)
        downscaled = self.processor.downscale_image(raw_bytes)

        with Image.open(io.BytesIO(downscaled)) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertEqual(img.mode, "RGB")
            self.assertLessEqual(max(img.size), 1024)
            # Width was 2000, height was 1000 (2:1 aspect ratio)
            self.assertEqual(img.size, (1024, 512))

    def test_small_image_retains_dimensions(self) -> None:
        raw_bytes = self._create_synthetic_image(300, 200)
        downscaled = self.processor.downscale_image(raw_bytes)

        with Image.open(io.BytesIO(downscaled)) as img:
            self.assertEqual(img.size, (300, 200))

    def test_rgba_to_rgb_conversion(self) -> None:
        raw_bytes = self._create_synthetic_image(400, 400, mode="RGBA", format_name="PNG")
        downscaled = self.processor.downscale_image(raw_bytes)

        with Image.open(io.BytesIO(downscaled)) as img:
            self.assertEqual(img.mode, "RGB")

    def test_animated_gif_extracts_first_frame(self) -> None:
        # Create a 2-frame GIF
        frame1 = Image.new("RGB", (200, 200), color=(255, 0, 0))
        frame2 = Image.new("RGB", (200, 200), color=(0, 255, 0))
        buf = io.BytesIO()
        frame1.save(
            buf,
            format="GIF",
            save_all=True,
            append_images=[frame2],
            duration=100,
            loop=0,
        )
        gif_bytes = buf.getvalue()

        result = self.processor.downscale_image(gif_bytes)
        with Image.open(io.BytesIO(result)) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertEqual(img.size, (200, 200))

    def test_corrupt_bytes_raise_error(self) -> None:
        with self.assertRaises(ImageProcessingError):
            self.processor.downscale_image(b"not-an-image-payload")

    def test_empty_bytes_raise_error(self) -> None:
        with self.assertRaises(ImageProcessingError):
            self.processor.downscale_image(b"")

    async def test_async_process_image(self) -> None:
        raw_bytes = self._create_synthetic_image(500, 500)
        result = await self.processor.process_image(raw_bytes)
        self.assertTrue(len(result) > 0)


class TestVisionService(unittest.IsolatedAsyncioTestCase):
    async def test_describe_image_success(self) -> None:
        config = BotConfig(
            discord_token="fake_token",
            vision_enabled=True,
            vision_model="moondream",
        )
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "message": {
                "content": "<think>Analyzing image...</think>A cute calico cat sleeping on a wooden table."
            }
        }

        service = VisionService(config=config, client=mock_client)
        description = await service.describe_image(b"fake_image_bytes")

        self.assertEqual(description, "A cute calico cat sleeping on a wooden table.")
        mock_client.chat.assert_awaited_once()

        # Check that payload contains base64 image and prompt
        call_kwargs = mock_client.chat.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "moondream")
        messages = call_kwargs["messages"]
        self.assertEqual(len(messages), 1)
        self.assertIn("images", messages[0])
        self.assertEqual(len(messages[0]["images"]), 1)
        self.assertEqual(call_kwargs["options"]["num_predict"], 1024)
        self.assertEqual(call_kwargs["options"]["num_ctx"], 4096)

    async def test_describe_image_context_aware_query(self) -> None:
        config = BotConfig(
            discord_token="fake_token",
            vision_enabled=True,
            vision_model="qwen2.5vl:3b",
            vision_num_predict=1024,
        )
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "message": {
                "content": "Problem 2 asks to solve for x in 2x + 5 = 15. The answer is x = 5."
            }
        }

        service = VisionService(config=config, client=mock_client)
        user_query = "Please help me solve problem 2 on this homework"
        description = await service.describe_image(b"fake_image_bytes", user_query=user_query)

        self.assertIn("Problem 2", description)
        call_kwargs = mock_client.chat.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "qwen2.5vl:3b")
        prompt_content = call_kwargs["messages"][0]["content"]
        self.assertIn(user_query, prompt_content)
        self.assertIn("transcribe", prompt_content)
        self.assertEqual(call_kwargs["options"]["num_predict"], 1024)

    async def test_describe_image_when_disabled(self) -> None:

        config = BotConfig(
            discord_token="fake_token",
            vision_enabled=False,
        )
        mock_client = AsyncMock()
        service = VisionService(config=config, client=mock_client)

        description = await service.describe_image(b"fake_image_bytes")
        self.assertEqual(description, "")
        mock_client.chat.assert_not_awaited()

    async def test_describe_image_error_raises_exception(self) -> None:
        config = BotConfig(
            discord_token="fake_token",
            vision_enabled=True,
        )
        mock_client = AsyncMock()
        mock_client.chat.side_effect = RuntimeError("Ollama connection refused")

        service = VisionService(config=config, client=mock_client)
        with self.assertRaises(VisionServiceError):
            await service.describe_image(b"fake_image_bytes")


class TestTextHistoryWithVision(unittest.TestCase):
    def test_text_conversion_in_memory_manager(self) -> None:
        memory = MemoryManager(max_turns=3)

        # Simulate turn 1 with image converted to text
        image_desc = "A small orange kitten tangled in yarn."
        user_prompt = "Look at what she did!"
        formatted_prompt = f"User1: [Attached Image: {image_desc}]\n{user_prompt}"
        assistant_reply = "べつに, kittens always make a mess. Don't let her chew the string..."

        memory.add_turn(
            guild_id=1,
            channel_id=100,
            user_content=formatted_prompt,
            assistant_content=assistant_reply,
        )

        # Verify memory contains only string text
        history = memory.get_history(guild_id=1, channel_id=100)
        self.assertEqual(len(history), 2)
        self.assertIsInstance(history[0]["content"], str)
        self.assertIn("[Attached Image: A small orange kitten tangled in yarn.]", history[0]["content"])
        self.assertNotIn("images", history[0])  # No image binary in history

        # Simulate turn 2 (follow-up query referencing the image)
        turn2_prompt = "User1: Is she okay though?"
        turn2_reply = "Of course she's fine, baka! Just untangle her already."
        memory.add_turn(
            guild_id=1,
            channel_id=100,
            user_content=turn2_prompt,
            assistant_content=turn2_reply,
        )

        updated_history = memory.get_history(guild_id=1, channel_id=100)
        self.assertEqual(len(updated_history), 4)
        # Turn 1's image description remains in text form
        self.assertIn("tangled in yarn", updated_history[0]["content"])


if __name__ == "__main__":
    unittest.main()
