from __future__ import annotations

import os
import io
import base64
import mimetypes
from typing import Optional, List, Union

from groq import Groq
from openai import OpenAI
from PIL import Image

from ocr.pdf_to_images import pdf_to_stitched_images

ImageInput = Union[str, io.BytesIO, bytes, Image.Image]  # path | BytesIO | raw bytes | PIL.Image | base64/data-URI str
PdfInput = Union[str, os.PathLike, bytes, io.BytesIO]





class groqLLM:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq LLM client.

        Args:
            api_key (str, optional): Groq API key. If None, uses GROQ_API_KEY from environment.
            model (str, optional): Default model name to use for inference.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not provided or found in environment variable GROQ_API_KEY.")
        self.client = Groq(api_key=self.api_key)
        self.model = model

    # ---------- image helpers ----------

    @staticmethod
    def _encode_image_file(path: str) -> tuple[str, str]:
        with open(path, "rb") as f:
            data = f.read()
        mime, _ = mimetypes.guess_type(path)
        if mime is None:
            mime = "image/png"
        return base64.b64encode(data).decode("utf-8"), mime

    @staticmethod
    def _pil_to_b64(img: Image.Image, format: str = "PNG") -> tuple[str, str]:
        buf = io.BytesIO()
        img.save(buf, format=format)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8"), f"image/{format.lower()}"

    @staticmethod
    def _to_b64_and_mime(img: ImageInput, default_mime: str = "image/png") -> tuple[str, str]:
        """
        Normalize various image inputs to (base64_string, mime).
        Accepts: path, BytesIO, bytes, PIL.Image, base64 string, or data URI.
        """
        if isinstance(img, Image.Image):
            return groqLLM._pil_to_b64(img, "PNG")
        if isinstance(img, io.BytesIO):
            img.seek(0)
            return base64.b64encode(img.read()).decode("utf-8"), default_mime
        if isinstance(img, (bytes, bytearray)):
            return base64.b64encode(bytes(img)).decode("utf-8"), default_mime
        if isinstance(img, str):
            s = img.strip()
            if s.startswith("data:image/"):
                # Already a data URI -> parse out mime and b64
                head, b64 = s.split(",", 1)
                mime = head.split(";")[0].split(":")[1]
                return b64, mime
            if os.path.exists(s):
                return groqLLM._encode_image_file(s)
            # Assume bare base64 string
            return s, default_mime
        raise TypeError("Unsupported image type. Use path | BytesIO | bytes | PIL.Image | base64/data-URI str.")

    # ---------- core infer ----------

    def infer(
        self,
        prompt: Optional[str] = None,
        *,
        images: Optional[List[ImageInput]] = None,
        pdfs: Optional[List[PdfInput]] = None,
        model: Optional[str] = None,
        # PDF rendering/stitching options forwarded to pdf_to_stitched_images
        pdf_options: Optional[dict] = None,
        **kwargs
    ) -> str:
        """
        Flexible multimodal inference.

        You can pass:
          - prompt only
          - images only (paths, BytesIO, bytes, PIL.Image, base64 string, or data URI)
          - pdfs only (paths, bytes, BytesIO) -> auto converted to stitched images
          - any combination (images+text, pdfs+text, images+pdfs+text)

        Args:
            prompt: Optional text to include.
            images: Optional list of image-like inputs.
            pdfs: Optional list of PDFs; each converted into 1+ stitched images.
            model: Optional model override.
            pdf_options: Dict of options for `pdf_to_stitched_images`, e.g.:
                {
                    "canvas_w": 1500,
                    "divider_px": 3,
                    "render_zoom": 2.0,
                    "bg_color": (255,255,255),
                    "divider_color": (0,0,0),
                    "output_format": "PNG"
                }

        Returns:
            str: The model's textual response.
        """
        chosen_model = model or self.model
        content = []

        # 1) Text first (if any)
        if prompt and prompt.strip():
            content.append({"type": "text", "text": prompt})

        # 2) Aggregate all images (direct + from PDFs)
        normalized_images: List[ImageInput] = []

        if images:
            normalized_images.extend(images)

        if pdfs:
            opts = {"return_as_object": True, "output_format": "PNG"}
            if pdf_options:
                # protect from forcing disk writes
                opts.update({k: v for k, v in pdf_options.items() if k != "output_dir"})
            for pdf in pdfs:
                # Convert each pdf to list of in-memory PNGs
                buffers = pdf_to_stitched_images(pdf, **opts)
                if buffers:
                    normalized_images.extend(buffers)

        if not content and not normalized_images:
            raise ValueError("Nothing to send. Provide at least one of: prompt, images, or pdfs.")

        # 3) Add images to content as data URLs
        for img in normalized_images:
            b64, mime = self._to_b64_and_mime(img, default_mime="image/png")
            data_url = f"data:{mime};base64,{b64}"
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })

        # 4) Send to Groq
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model=chosen_model,
            **kwargs
        )
        return chat_completion.choices[0].message.content



class openaiLLM:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini", reasoning_model: bool = True):
        """
        Initialize OpenAI LLM client.

        Args:
            api_key (str, optional): OpenAI API key. If None, uses OPENAI_API_KEY from environment.
            model (str, optional): Default model name to use for inference.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided or found in environment variable OPENAI_API_KEY.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.reasoning_model = reasoning_model
        self.current_input_token_usage = 0
        self.total_input_token_usage = 0
        self.current_output_token_usage = 0
        self.total_output_token_usage = 0

    # ---------- image helpers ----------

    @staticmethod
    def _encode_image_file(path: str) -> tuple[str, str]:
        with open(path, "rb") as f:
            data = f.read()
        mime, _ = mimetypes.guess_type(path)
        if mime is None:
            mime = "image/png"
        return base64.b64encode(data).decode("utf-8"), mime

    @staticmethod
    def _pil_to_b64(img: Image.Image, format: str = "PNG") -> tuple[str, str]:
        buf = io.BytesIO()
        img.save(buf, format=format)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8"), f"image/{format.lower()}"

    @staticmethod
    def _to_b64_and_mime(img: ImageInput, default_mime: str = "image/png") -> tuple[str, str]:
        """
        Normalize various image inputs to (base64_string, mime).
        Accepts: path, BytesIO, bytes, PIL.Image, base64 string, or data URI.
        """
        if isinstance(img, Image.Image):
            return openaiLLM._pil_to_b64(img, "PNG")
        if isinstance(img, io.BytesIO):
            img.seek(0)
            return base64.b64encode(img.read()).decode("utf-8"), default_mime
        if isinstance(img, (bytes, bytearray)):
            return base64.b64encode(bytes(img)).decode("utf-8"), default_mime
        if isinstance(img, str):
            s = img.strip()
            if s.startswith("data:image/"):
                head, b64 = s.split(",", 1)
                mime = head.split(";")[0].split(":")[1]
                return b64, mime
            if os.path.exists(s):
                return openaiLLM._encode_image_file(s)
            # Assume bare base64 string
            return s, default_mime
        raise TypeError("Unsupported image type. Use path | BytesIO | bytes | PIL.Image | base64/data-URI str.")

    def _update_usage_counters(self, response) -> None:
        """
        Update per-call and cumulative token usage counters.

        - Input tokens  = prompt_tokens
        - Output tokens = completion_tokens + reasoning_tokens (if available)
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            # e.g. some streaming cases or older responses
            self.current_input_token_usage = 0
            self.current_output_token_usage = 0
            return

        # Base fields (always try to exist per docs)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        # --- Reasoning tokens handling ---
        reasoning_tokens = 0

        # 1) Newer APIs: nested completion_tokens_details
        ctd = getattr(usage, "completion_tokens_details", None)
        if ctd is not None:
            # If it's a Pydantic model (most likely)
            if hasattr(ctd, "reasoning_tokens"):
                reasoning_tokens = getattr(ctd, "reasoning_tokens", 0) or 0
            # If some client returns it as plain dict
            elif isinstance(ctd, dict):
                reasoning_tokens = ctd.get("reasoning_tokens", 0) or 0

        # 2) Some variants expose reasoning_tokens directly on usage
        if not reasoning_tokens and hasattr(usage, "reasoning_tokens"):
            reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0

        # Normalize to ints
        input_tokens = int(prompt_tokens)
        output_tokens = int(completion_tokens + reasoning_tokens)

        # Set "current" usage (this call)
        self.current_input_token_usage = input_tokens
        self.current_output_token_usage = output_tokens

        # Accumulate "total" usage
        self.total_input_token_usage += input_tokens
        self.total_output_token_usage += output_tokens

    # ---------- core infer ----------

    def infer(
        self,
        prompt: Optional[str] = None,
        *,
        images: Optional[List[ImageInput]] = None,
        pdfs: Optional[List[PdfInput]] = None,
        model: Optional[str] = None,
        pdf_options: Optional[dict] = None,
        reasoning_effort: Optional[str] = "minimal",
        **kwargs
    ) -> str:
        """
        Flexible multimodal inference for OpenAI GPT models.

        You can pass:
          - prompt only
          - images only (paths, BytesIO, bytes, PIL.Image, base64 string, or data URI)
          - pdfs only (paths, bytes, BytesIO) -> auto converted to stitched images
          - any combination (images+text, pdfs+text, images+pdfs+text)
          - reasoning_effort ('minimal', 'low', 'medium', 'high')
        """
        chosen_model = model or self.model
        content = []

        # 1) Add text first
        if prompt and prompt.strip():
            content.append({"type": "text", "text": prompt})

        # 2) Gather images (direct + from PDFs)
        normalized_images: List[ImageInput] = []

        if images:
            normalized_images.extend(images)

        if pdfs:
            opts = {"return_as_object": True, "output_format": "PNG"}
            if pdf_options:
                opts.update({k: v for k, v in pdf_options.items() if k != "output_dir"})
            for pdf in pdfs:
                buffers = pdf_to_stitched_images(pdf, **opts)
                if buffers:
                    normalized_images.extend(buffers)

        if not content and not normalized_images:
            raise ValueError("Nothing to send. Provide at least one of: prompt, images, or pdfs.")

        # 3) Convert images to data URLs
        for img in normalized_images:
            b64, mime = self._to_b64_and_mime(img, default_mime="image/png")
            data_url = f"data:{mime};base64,{b64}"
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })

        # 4) Send to OpenAI
        if self.reasoning_model:
            response = self.client.chat.completions.create(
                model=chosen_model,
                reasoning_effort=reasoning_effort,
                messages=[{"role": "user", "content": content}],
                **kwargs
            )
        else:
            response = self.client.chat.completions.create(
                model=chosen_model,
                messages=[{"role": "user", "content": content}],
                **kwargs
            )

        # 5) Update token usage counters
        self._update_usage_counters(response)


        return response.choices[0].message.content
