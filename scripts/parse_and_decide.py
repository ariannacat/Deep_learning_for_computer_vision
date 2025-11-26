
#!/usr/bin/env python3

"""
One-shot pipeline runner:
Screenshot → parse → recognize Pokémon → decide best move.
"""

import argparse
import json
import cv2

from pokeai.screengrab_parser import extract_from_screenshot
from pokeai.pipeline import run
from pokeai.config import load_config

def main():
    ap = argparse.ArgumentParser(
        description="Parse a battle screenshot and compute the best move."
    )

    ap.add_argument("--image", required=True, help="Path to screenshot (PNG/JPG)")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="Path to configuration file")

    args = ap.parse_args()

    # --- Load image ---
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Error: cannot read image {args.image}")

    # --- First: your raw parser (kept untouched) ---
    raw_parse = extract_from_screenshot(img)

    print("\n=== RAW PARSE RESULT ===")
    print(json.dumps(raw_parse, indent=2))

    # --- Second: full pipeline (recognize Pokémon + decide best move) ---
    cfg = load_config(args.config)
    parse_result, decision = run(args.image, args.config)

    print("\n=== PIPELINE PARSE RESULT (normalized) ===")
    print(json.dumps(parse_result.to_dict(), indent=2))

    print("\n=== DECISION ===")
    print(json.dumps(decision.to_dict(), indent=2))

if __name__ == "__main__":
    main()

