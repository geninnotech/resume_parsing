from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, Iterable, Callable
import io

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


PdfSource = Union[str, Path, bytes, io.BytesIO]
PreprocessFn = Callable[[Image.Image, int], Image.Image]  # (image, page_index) -> image


def _open_pdf(pdf_source: PdfSource) -> fitz.Document:
    """
    Open a PDF from a path, bytes, or BytesIO.
    """
    if isinstance(pdf_source, (str, Path)):
        p = Path(pdf_source)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
        return fitz.open(p)
    if isinstance(pdf_source, io.BytesIO):
        data = pdf_source.getvalue()
        return fitz.open(stream=data, filetype="pdf")
    if isinstance(pdf_source, (bytes, bytearray)):
        return fitz.open(stream=bytes(pdf_source), filetype="pdf")
    raise TypeError("Unsupported pdf_source type. Use str/Path/bytes/BytesIO.")


def ocr_pdf_to_text(
    pdf_source: PdfSource,
    *,
    lang: str = "eng",
    dpi: int = 300,
    page_separator: str = "\n\n" + ("-" * 40) + "\n\n",
    pages: Optional[Iterable[int]] = None,
    progress: bool = True,
    preprocess: Optional[PreprocessFn] = None,
    tesseract_config: Optional[str] = None,
    psm: Optional[int] = None,
    oem: Optional[int] = None,
    timeout: Optional[int] = None,
    return_pages: bool = False,
) -> Union[str, List[str]]:
    """
    OCR a PDF and return recognized text.

    Args:
        pdf_source: Path, bytes, or BytesIO for the input PDF.
        lang: Tesseract language code(s), e.g. "eng", "eng+hin".
        dpi: Render DPI (higher -> sharper OCR, slower).
        page_separator: Separator used when joining page outputs.
        pages: Optional iterable of 0-based page indices to OCR. If None, OCR all pages.
        progress: If True, prints simple progress logs.
        preprocess: Optional function(img: PIL.Image, page_index: int) -> PIL.Image to modify the raster before OCR.
        tesseract_config: Extra config string for Tesseract, e.g. "--dpi 300".
        psm: Tesseract page segmentation mode (int). Appended as "--psm <psm>".
        oem: Tesseract OCR engine mode (int). Appended as "--oem <oem>".
        timeout: Optional timeout (seconds) for pytesseract.image_to_string.
        return_pages: If True, return a list of per-page strings; else a single string joined by page_separator.

    Returns:
        str (default) with all pages joined by `page_separator`,
        or List[str] if return_pages=True.
    """
    # Build tesseract config
    config_parts: List[str] = []
    if tesseract_config:
        config_parts.append(tesseract_config.strip())
    if psm is not None:
        config_parts.append(f"--psm {psm}")
    if oem is not None:
        config_parts.append(f"--oem {oem}")
    config_str = " ".join(part for part in config_parts if part)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    texts: List[str] = []

    with _open_pdf(pdf_source) as doc:
        # Decide which pages to process
        if pages is None:
            page_indices = range(len(doc))
        else:
            page_indices = list(pages)

        for i, page_idx in enumerate(page_indices):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
            # Use PPM bytes to avoid format lossiness and keep color
            pil_img = Image.open(io.BytesIO(pix.tobytes("ppm")))

            # Optional preprocessing hook
            if preprocess is not None:
                pil_img = preprocess(pil_img, page_idx)

            # OCR
            txt = pytesseract.image_to_string(
                pil_img,
                lang=lang,
                config=config_str if config_str else None,
                timeout=timeout
            )

            texts.append(txt)
            if progress:
                print(f"OCR page {i+1}/{len(page_indices)} (doc page {page_idx+1}): {len(txt)} chars")

    if return_pages:
        return texts
    return page_separator.join(texts)
