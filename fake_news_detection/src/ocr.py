"""Image → text extraction for social-media screenshots.

Primary: EasyOCR (pip-installable, supports Indonesian + English out of the box).
Fallback: pytesseract if EasyOCR is not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union


def extract_text_from_image(
    image: Union[str, Path, bytes],
    languages: Optional[List[str]] = None,
    gpu: bool = False,
    engine: str = "auto",   # "auto" | "easyocr" | "tesseract"
) -> str:
    """Return all detected text in `image`, joined by newlines.

    Parameters
    ----------
    image : path (str / Path) OR raw bytes
    languages : list of EasyOCR language codes, default ['id', 'en']
    gpu : let EasyOCR use GPU if available
    engine : force a specific OCR engine, or 'auto' to try EasyOCR first
    """
    languages = languages or ["id", "en"]

    if engine in ("auto", "easyocr"):
        try:
            return _easyocr(image, languages, gpu)
        except ImportError:
            if engine == "easyocr":
                raise
        except Exception:
            if engine == "easyocr":
                raise

    return _tesseract(image)


# ---------------------------------------------------------------------------

def _easyocr(image, languages, gpu) -> str:
    import easyocr

    reader = easyocr.Reader(languages, gpu=gpu, verbose=False)

    if isinstance(image, (str, Path)):
        result = reader.readtext(str(image), detail=0, paragraph=True)
    else:
        # raw bytes
        import io
        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(image)).convert("RGB")
        arr = np.array(img)
        result = reader.readtext(arr, detail=0, paragraph=True)

    return "\n".join(s.strip() for s in result if s.strip())


def _tesseract(image) -> str:
    import pytesseract
    from PIL import Image
    import io

    if isinstance(image, (str, Path)):
        img = Image.open(str(image))
    else:
        img = Image.open(io.BytesIO(image))
    # tesseract language packs: 'ind' + 'eng' = Indonesian + English
    return pytesseract.image_to_string(img, lang="ind+eng").strip()
