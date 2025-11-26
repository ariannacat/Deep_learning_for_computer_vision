"""
Screenshot parser for Pokémon Showdown-style battle UI.

Given a full screenshot (BGR numpy array), this module:
- crops own and opponent sprites
- auto-detects HP bars and estimates HP%
- extracts the 4 move names from the bottom panel via OCR

Main entry:
    extract_from_screenshot(bgr: np.ndarray) -> dict
"""


from __future__ import annotations

from typing import Tuple, List, Dict, Any

import numpy as np
import cv2

from .ocr import ocr_text_moves, clean_move_text, correct_with_dictionary


# =========================
# CONFIG (percentual ROIs)
# =========================
# ROIs are in *relative* coords [0..1] w.r.t. the ORIGINAL IMAGE.
# These values are tuned on a 1294x954 screenshot but generalize well.

ROI: Dict[str, Tuple[float, float, float, float]] = {
    # Our and opponent sprites
    "own_sprite": (0.18, 0.35, 0.45, 0.80),
    "opp_sprite": (0.50, 0.10, 0.83, 0.45),

    # Moves panel (bottom area with 4 buttons)
    "moves_panel": (0.05, 0.86, 0.95, 0.98),
}

# Move dictionary loaded from CSV via io_utils.
# It must be set using set_move_dictionary() before using extract_from_screenshot.
MOVE_DICTIONARY: List[str] = []

def set_move_dictionary(moves: List[str]) -> None:
    """
    Setter to define the canonical list of move names used to correct OCR output.
    Call this once in the pipeline before parsing screenshots.
    """
    global MOVE_DICTIONARY
    MOVE_DICTIONARY = sorted(set(moves))


# =========================
# Geometry helpers
# =========================

def to_abs(box_rel: Tuple[float, float, float, float], W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box_rel
    return int(x0*W), int(y0*H), int(x1*W), int(y1*H)

def crop_rel(img: np.ndarray, box_rel: Tuple[float, float, float, float]) -> np.ndarray:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = to_abs(box_rel, W, H)
    return img[y0:y1, x0:x1]

# =========================
# HP BAR auto-detect (color)
# =========================

def find_hp_bar_and_percent(bgr: np.ndarray, side: str) -> Dict[str, Any] | None:
    """
    Automatically search for a long, thin HP bar using HSV color.

    side: 'left' or 'right' to limit search to half the image.

    Returns a dict:
        {
            "bbox_abs": (x, y, w, h),
            "hp_percent": float | None,
            "color": "green" | "yellow" | "red",
        }
    or None if no candidate is found.
    """
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Color masks (tunable)
    lower_g = np.array([35, 30, 35]); upper_g = np.array([85, 255, 255])
    lower_y = np.array([20, 30, 35]); upper_y = np.array([34, 255, 255])
    lower_r1 = np.array([0, 30, 35]);  upper_r1 = np.array([10, 255, 255])
    lower_r2 = np.array([170, 30, 35]);upper_r2 = np.array([180, 255, 255])

    m_g = cv2.inRange(hsv, lower_g, upper_g)
    m_y = cv2.inRange(hsv, lower_y, upper_y)
    m_r = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1),
                         cv2.inRange(hsv, lower_r2, upper_r2))
    colored = cv2.bitwise_or(cv2.bitwise_or(m_g, m_y), m_r)

    # Restrict to left or right half 
    if side == "left":
        colored[:, int(W*0.55):] = 0
    else:
        colored[:, :int(W*0.45)] = 0

    # Look for horizontal blobs 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    merged = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        ar = w / max(1,h)
        if w > W*0.12 and 5 <= h <= 22 and ar > 5:  # larga e sottile
            cand.append((w*h, (x,y,w,h)))
    if not cand:
        return None
 
    _, (x,y,w,h) = max(cand, key=lambda z: z[0])

    # Estimate HP% = length of colored part / total bar length
    roi = bgr[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mc = cv2.inRange(hsv_roi, lower_g, upper_g)
    mc = cv2.bitwise_or(mc, cv2.inRange(hsv_roi, lower_y, upper_y))
    mc = cv2.bitwise_or(mc, cv2.inRange(hsv_roi, lower_r1, upper_r1))
    mc = cv2.bitwise_or(mc, cv2.inRange(hsv_roi, lower_r2, upper_r2))

    proj = mc.sum(axis=0)  # somma colonne
    xs = np.where(proj > 0)[0]
    if len(xs) < 3:
        hp_pct = None
    else:
        col_len = xs.max() - xs.min() + 1
        hp_pct = float(np.clip(100.0 * col_len / w, 0, 100))

    # Colore dominante
    g_count = int((mc > 0).sum() if (cv2.countNonZero(cv2.inRange(hsv_roi, lower_g, upper_g))>0) else 0)
    y_count = cv2.countNonZero(cv2.inRange(hsv_roi, lower_y, upper_y))
    r_count = (cv2.countNonZero(cv2.inRange(hsv_roi, lower_r1, upper_r1)) +
               cv2.countNonZero(cv2.inRange(hsv_roi, lower_r2, upper_r2)))
    dom = "green"
    if max(y_count, r_count) > g_count:
        dom = "yellow" if y_count >= r_count else "red"

    return {
        "bbox_abs": (x,y,w,h),
        "hp_percent": None if hp_pct is None else round(hp_pct, 1),
        "color": dom
    }


# =========================
# Moves panel helpers
# =========================


def split_moves_panel(panel_bgr: np.ndarray, pad: int = 8) -> List[np.ndarray]:
    """
    Bottom moves panel: 4 buttons in 1 row (1x4).

    Returns, for each tile, the central band where the move name sits
    (excluding the type icon on the left and PP area on the right).
    """
    H, W = panel_bgr.shape[:2]

    # Panel margins
    xL = pad
    xR = W - pad
    yT = pad
    yB = H - pad

    # Width of each move tile
    tile_w = (xR - xL) // 4
    tiles: List[np.ndarray] = []
    for i in range(4):
        x0 = xL + i * tile_w
        x1 = x0 + tile_w
        tile = panel_bgr[yT:yB, x0:x1]

        # Central band of the name (cut out type badge and PP)
        th, tw = tile.shape[:2]
        y0 = int(th * 0.30)   # ~30% to ~65% vertically
        y1 = int(th * 0.65)
        x0b = int(tw * 0.16)  # ~16% to ~86% horizontally
        x1b = int(tw * 0.86)
        name_band = tile[y0:y1, x0b:x1b]
        tiles.append(name_band)

    return tiles

# =========================
# MAIN extraction
# =========================

def extract_from_screenshot(bgr: np.ndarray) -> Dict[str, Any]:
    """
    Main entry point: parse a BGR screenshot into structured information.

    Returns a dict:
        {
            "sprites": {
                "own_sprite_file": ...,
                "opp_sprite_file": ...,
                "own_sprite_roi": ...,
                "opp_sprite_roi": ...
            },
            "hp": {
                "ours": {...} | None,
                "opponent": {...} | None,
            },
            "moves": [move1, move2, move3, move4]
        }
    """
    H, W = bgr.shape[:2]

    # 1) Sprite crops (only Pokémon bodies, no name labels)
    own_sprite = crop_rel(bgr, ROI["own_sprite"])
    opp_sprite = crop_rel(bgr, ROI["opp_sprite"])

    # Save crops to files (your recognition model can load these)
    cv2.imwrite("crop_own_sprite.png", own_sprite)
    cv2.imwrite("crop_opp_sprite.png", opp_sprite)

    # 2) HP bar auto-detect (left = ours, right = opponent)
    hp_left = find_hp_bar_and_percent(bgr, side="left")
    hp_right = find_hp_bar_and_percent(bgr, side="right")

    # 3) Moves (bottom 1x4)
    moves_panel = crop_rel(bgr, ROI["moves_panel"])
    tiles = split_moves_panel(moves_panel, pad=8)

    moves: List[str] = []
    for i, t in enumerate(tiles, 1):
        m = ocr_text_moves(t)
        m = clean_move_text(m)
        m = correct_with_dictionary(m, MOVE_DICTIONARY, cutoff=0.45)
        moves.append(m)
        # Optional debug: save each move crop
        cv2.imwrite(f"debug_move_{i}.png", t)

    return {
        "sprites": {
            "own_sprite_file": "crop_own_sprite.png",
            "opp_sprite_file": "crop_opp_sprite.png",
            "own_sprite_roi": ROI["own_sprite"],
            "opp_sprite_roi": ROI["opp_sprite"],
        },
        "hp": {
            "ours": hp_left,      # {bbox_abs, hp_percent, color} or None
            "opponent": hp_right,
        },
        "moves": moves,
    }    
