from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional, Union
import io
import fitz  # PyMuPDF
from PIL import Image, ImageDraw

PdfSource = Union[str, Path, bytes, io.BytesIO]


def _resize_to_width(img: Image.Image, target_w: int) -> Image.Image:
    """
    Resize `img` to fit exactly `target_w` in width, preserving aspect ratio.
    Returned image height will vary according to aspect ratio.
    """
    w, h = img.size
    if w <= 0:
        return img
    scale = target_w / w
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _open_pdf(pdf_source: PdfSource) -> fitz.Document:
    """
    Open a PDF from a path, bytes, or BytesIO.
    """
    if isinstance(pdf_source, (str, Path)):
        return fitz.open(Path(pdf_source))
    if isinstance(pdf_source, io.BytesIO):
        data = pdf_source.getvalue()
        return fitz.open(stream=data, filetype="pdf")
    if isinstance(pdf_source, (bytes, bytearray)):
        return fitz.open(stream=bytes(pdf_source), filetype="pdf")
    raise TypeError("Unsupported pdf_source type. Use str/Path/bytes/BytesIO.")


# def pdf_to_stitched_images(
#     pdf_source: PdfSource,
#     output_dir: Optional[str] = None,
#     *,
#     canvas_w: int = 1000,
#     divider_px: int = 3,
#     bg_color: Tuple[int, int, int] = (255, 255, 255),
#     divider_color: Tuple[int, int, int] = (0, 0, 0),
#     render_zoom: float = 2.0,
#     return_as_object: bool = False,
#     output_format: str = "PNG",
#     max_pages: Optional[int] = 6,
# ) -> Optional[List[io.BytesIO]]:
#     """
#     Render a PDF into stitched images (two pages per output, last lone page single).

#     If `return_as_object` is False (default):
#         - Saves files to `output_dir` as out_XXX.png (or chosen format).
#         - Returns None.

#     If `return_as_object` is True:
#         - Returns a list of io.BytesIO objects (in-memory images).
#         - Does NOT write anything to disk and ignores `output_dir`.

#     The returned BytesIO objects are compatible with:
#         base64.b64encode(buf.read()).decode("utf-8")
#     """
#     if not return_as_object:
#         if output_dir is None:
#             raise ValueError("`output_dir` is required when `return_as_object` is False.")
#         Path(output_dir).mkdir(parents=True, exist_ok=True)

#     # Open PDF and render pages
#     doc = _open_pdf(pdf_source)
#     pages: List[Image.Image] = []
#     mat = fitz.Matrix(render_zoom, render_zoom)
#     try:
#         for i in range(len(doc)):
#             page = doc[i]
#             pix = page.get_pixmap(matrix=mat, alpha=False)
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             pages.append(img)
#     finally:
#         doc.close()

#     # Compute left/right widths
#     left_w = (canvas_w - divider_px) // 2
#     right_w = canvas_w - divider_px - left_w
#     out_index = 1
#     i = 0

#     # If returning objects, collect buffers here
#     buffers: List[io.BytesIO] = []

#     while i < len(pages):
#         if i + 1 < len(pages):
#             # Pair: stitch two pages
#             left_img = _resize_to_width(pages[i], left_w)
#             right_img = _resize_to_width(pages[i + 1], right_w)

#             # Make canvas height equal to the taller of the two images
#             canvas_h = max(left_img.height, right_img.height)
#             combined = Image.new("RGB", (canvas_w, canvas_h), bg_color)

#             # Paste left at top
#             combined.paste(left_img, (0, 0))

#             # Divider
#             draw = ImageDraw.Draw(combined)
#             x0 = left_w
#             draw.rectangle([x0, 0, x0 + divider_px - 1, canvas_h - 1], fill=divider_color)

#             # Paste right at top (x offset accounts for divider)
#             combined.paste(right_img, (left_w + divider_px, 0))

#             if return_as_object:
#                 buf = io.BytesIO()
#                 combined.save(buf, format=output_format)
#                 buf.seek(0)
#                 buffers.append(buf)
#             else:
#                 out_name = Path(output_dir) / f"out_{out_index:03d}.{output_format.lower()}"
#                 combined.save(out_name, output_format)
#             out_index += 1
#             i += 2
#         else:
#             # Single last page
#             single_img = _resize_to_width(pages[i], canvas_w)
#             canvas_h = single_img.height
#             single = Image.new("RGB", (canvas_w, canvas_h), bg_color)
#             single.paste(single_img, (0, 0))

#             if return_as_object:
#                 buf = io.BytesIO()
#                 single.save(buf, format=output_format)
#                 buf.seek(0)
#                 buffers.append(buf)
#             else:
#                 out_name = Path(output_dir) / f"out_{out_index:03d}.{output_format.lower()}"
#                 single.save(out_name, output_format)
#             out_index += 1
#             i += 1

#     return buffers if return_as_object else None


def pdf_to_stitched_images(
    pdf_source: PdfSource,
    output_dir: Optional[str] = None,
    *,
    canvas_w: int = 1000,
    divider_px: int = 3,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    divider_color: Tuple[int, int, int] = (0, 0, 0),
    render_zoom: float = 2.0,
    return_as_object: bool = False,
    output_format: str = "PNG",
    max_pages: Optional[int] = 6,
) -> Optional[List[io.BytesIO]]:
    """
    Render a PDF into stitched images (two pages per output, last lone page single).

    Page limit: If `max_pages` is set (default 6), only the first `max_pages` pages
    are rendered; the rest are silently skipped. Set `max_pages=None` for no limit.

    If `return_as_object` is False (default):
        - Saves files to `output_dir` as out_XXX.png (or chosen format).
        - Returns None.

    If `return_as_object` is True:
        - Returns a list of io.BytesIO objects (in-memory images).
        - Does NOT write anything to disk and ignores `output_dir`.

    The returned BytesIO objects are compatible with:
        base64.b64encode(buf.read()).decode("utf-8")
    """
    if not return_as_object:
        if output_dir is None:
            raise ValueError("`output_dir` is required when `return_as_object` is False.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Open PDF and determine how many pages to render (silently truncate)
    doc = _open_pdf(pdf_source)
    pages: List[Image.Image] = []
    mat = fitz.Matrix(render_zoom, render_zoom)
    try:
        total_pages = len(doc)

        # Compute pages_to_render with safe coercion and no errors
        if max_pages is None:
            pages_to_render = total_pages
        else:
            try:
                limit = int(max_pages)
            except (TypeError, ValueError):
                limit = total_pages
            if limit < 0:
                limit = 0
            pages_to_render = min(total_pages, limit)

        # Render only the allowed pages
        for i in range(pages_to_render):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
    finally:
        doc.close()

    # Compute left/right widths
    left_w = (canvas_w - divider_px) // 2
    right_w = canvas_w - divider_px - left_w
    out_index = 1
    i = 0

    # If returning objects, collect buffers here
    buffers: List[io.BytesIO] = []

    while i < len(pages):
        if i + 1 < len(pages):
            # Pair: stitch two pages
            left_img = _resize_to_width(pages[i], left_w)
            right_img = _resize_to_width(pages[i + 1], right_w)

            # Make canvas height equal to the taller of the two images
            canvas_h = max(left_img.height, right_img.height)
            combined = Image.new("RGB", (canvas_w, canvas_h), bg_color)

            # Paste left at top
            combined.paste(left_img, (0, 0))

            # Divider
            draw = ImageDraw.Draw(combined)
            x0 = left_w
            draw.rectangle([x0, 0, x0 + divider_px - 1, canvas_h - 1], fill=divider_color)

            # Paste right at top (x offset accounts for divider)
            combined.paste(right_img, (left_w + divider_px, 0))

            if return_as_object:
                buf = io.BytesIO()
                combined.save(buf, format=output_format)
                buf.seek(0)
                buffers.append(buf)
            else:
                out_name = Path(output_dir) / f"out_{out_index:03d}.{output_format.lower()}"
                combined.save(out_name, output_format)
            out_index += 1
            i += 2
        else:
            # Single last page
            single_img = _resize_to_width(pages[i], canvas_w)
            canvas_h = single_img.height
            single = Image.new("RGB", (canvas_w, canvas_h), bg_color)
            single.paste(single_img, (0, 0))

            if return_as_object:
                buf = io.BytesIO()
                single.save(buf, format=output_format)
                buf.seek(0)
                buffers.append(buf)
            else:
                out_name = Path(output_dir) / f"out_{out_index:03d}.{output_format.lower()}"
                single.save(out_name, output_format)
            out_index += 1
            i += 1

    return buffers if return_as_object else None

