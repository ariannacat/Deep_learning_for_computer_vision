"""
OCR utilities for Pokémon battle screenshots.

These functions are adapted from your notebook:
- tuned preprocessing (upsample, sharpen/contrast, threshold)
- whitelisting
- cleaning / normalization
- dictionary-based correction
"""

from __future__ import annotations

from typing import List
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

# ============================================================
# OCR specialized for MOVE NAMES
# ============================================================

def ocr_text_moves(bgr: np.ndarray) -> str:
    """
    OCR optimized for Pokémon move names.

    Upsample + sharpen + adaptive threshold + whitelist.
    """
    h, w = bgr.shape[:2]
    scale = 3 if max(h, w) < 500 else 2
    bgr = cv2.resize(bgr, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    sharp = cv2.convertScaleAbs(sharp, alpha=1.4, beta=15)

    th = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 27, 10)
    th = cv2.medianBlur(th, 3)
    th = cv2.bitwise_not(th)

    config = '--oem 3 --psm 8 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -\'"'
    txt = pytesseract.image_to_string(Image.fromarray(th), lang="eng", config=config)
    txt = re.sub(r'[^A-Za-z \-\']', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt.title()


# ============================================================
# TEXT CLEANING & DICTIONARY CORRECTION
# ============================================================

def clean_move_text(t: str) -> str:
    """
    Clean and normalize OCR output for move names.
    Removes PP/Type/Level info and keeps only textual move name.
    """
    t = re.split(r'[|0-9/]+', t)[0]         
    t = re.sub(r'[^A-Za-z\' \-]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def correct_with_dictionary(text: str, candidates: List[str], cutoff: float = 0.7) -> str:
    """
    Correct OCR text by matching it against a list of candidate move names
    using difflib similarity.

    Returns the best match if similarity >= cutoff, else returns original text.
    """
    if not text or not candidates:
        return text
    import difflib
    m = difflib.get_close_matches(text, candidates, n=1, cutoff=cutoff)
    return m[0] if m else text
