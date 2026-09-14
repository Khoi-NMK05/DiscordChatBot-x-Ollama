from __future__ import annotations

import asyncio
import io
import logging
from typing import Tuple

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Raised when an image cannot be processed or decoded."""


class ImageProcessor:
    """
    Object-oriented utility for validating, downscaling, and converting images
    for vision LLM consumption. Prevents VRAM and visual token bloat.
    """

    SUPPORTED_MIME_TYPES: Tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
    )

    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
    )

    def __init__(self, max_dimension: int = 1024, quality: int = 85) -> None:
        self.max_dimension = max_dimension
        self.quality = quality

    def is_supported_mime_type(self, content_type: str | None) -> bool:
        """Validates if the given MIME content type is a supported image format."""
        if not content_type:
            return False
        clean_type = content_type.split(";")[0].strip().lower()
        return clean_type in self.SUPPORTED_MIME_TYPES

    def is_supported_filename(self, filename: str) -> bool:
        """Validates if the file extension is a supported image format."""
        lower_name = filename.lower()
        return any(lower_name.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)

    def downscale_image(self, raw_bytes: bytes) -> bytes:
        """
        Synchronously parses, normalizes orientation, downscales, and re-encodes
        an image into optimized JPEG bytes.

        - Extracts frame 0 if animated GIF.
        - Converts RGBA/transparency to RGB.
        - Scales down to max_dimension preserving aspect ratio.
        """
        if not raw_bytes:
            raise ImageProcessingError("Image byte payload is empty.")

        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                # If animated (e.g. GIF), select first frame
                if getattr(img, "is_animated", False):
                    img.seek(0)

                # Correct EXIF orientation if present
                img = ImageOps.exif_transpose(img)

                # Convert to RGB (handles RGBA, P palette, grayscale LA)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Downscale if larger than max_dimension
                width, height = img.size
                if max(width, height) > self.max_dimension:
                    scale = self.max_dimension / float(max(width, height))
                    new_width = max(1, int(width * scale))
                    new_height = max(1, int(height * scale))
                    resample_filter = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
                    img = img.resize((new_width, new_height), resample=resample_filter)
                    logger.debug(
                        "Downscaled image from %dx%d to %dx%d",
                        width,
                        height,
                        new_width,
                        new_height,
                    )

                output_buffer = io.BytesIO()
                img.save(
                    output_buffer,
                    format="JPEG",
                    quality=self.quality,
                    optimize=True,
                )
                return output_buffer.getvalue()

        except Exception as err:
            logger.error("Failed to process image bytes: %s", err)
            raise ImageProcessingError(f"Image processing failed: {err}") from err

    async def process_image(self, raw_bytes: bytes) -> bytes:
        """
        Asynchronously runs downscale_image in an executor thread to prevent
        blocking the asyncio event loop.
        """
        return await asyncio.to_thread(self.downscale_image, raw_bytes)
