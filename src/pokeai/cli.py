#!/usr/bin/env python3
"""
CLI entry point for the Pokémon Battle Advisor project.

This script exposes subcommands:
  - parse    : parse a screenshot (raw parser)
  - decide   : compute the best move (given a state JSON)
  - run      : full pipeline (parse + recognize + decide)

It is linked to the 'pokeai' terminal command via pyproject.toml:
    [project.scripts]
    pokeai = "cli.pokeai_cli:main"
"""

import argparse
import json
from pokeai.screengrab_parser import extract_from_screenshot
from pokeai.pipeline import run_pipeline
from pokeai.config import load_config
from pokeai.vision import predict_image


def main():
    parser = argparse.ArgumentParser(
        prog="pokeai",
        description="Pokémon Battle Advisor CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # ------------------ recognize ------------------
    p_recog = subparsers.add_parser("recognize", help="Recognize Pokémon only (sprites → names)")
    p_recog.add_argument("--image", required=True, help="Path to screenshot")
    p_recog.add_argument("--config", default="configs/default.yaml")
 
    # ------------------ parse ------------------
    p_parse = subparsers.add_parser("parse", help="Parse screenshot only (raw style)")
    p_parse.add_argument("--image", required=True, help="Path to screenshot")
    p_parse.add_argument("--config", default="configs/default.yaml")

    # ------------------ decide ------------------
    p_decide = subparsers.add_parser("decide", help="Compute best move from a JSON state")
    p_decide.add_argument("--state", required=True, help="Path to state JSON")
    p_decide.add_argument("--config", default="configs/default.yaml")

    # ------------------ run (full pipeline) ------------------
    p_run = subparsers.add_parser(
        "run",
        help="Full pipeline: screenshot → parse → recognize Pokémon → decide best move"
    )
    p_run.add_argument("--image", required=True)
    p_run.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()

    # --------------------------------------------------------
    # process commands
    # --------------------------------------------------------

    if args.command == "parse":
        import cv2
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Error: cannot read image {args.image}")

        result = extract_from_screenshot(img)
        print(json.dumps(result, indent=2))
        return

    elif args.command == "decide":
        with open(args.state) as f:
            state_data = json.load(f)

        cfg = load_config(args.config)

        # In the future, decision() will be inside pokeai.pipeline
        from pokeai.decision import advisor
        decision = advisor(state_data, cfg)

        print(json.dumps(decision.to_dict(), indent=2))
        return

    elif args.command == "run":
        cfg = load_config(args.config)
        from pokeai.pipeline import run_pipeline

        result = run_pipeline(args.image, cfg)

        print("\n=== PARSED STATE ===")
        print(json.dumps(result["parse"], indent=2))

        print("\n=== RECOGNITION ===")
        print(json.dumps(result["recognition"], indent=2))

        print("\n=== DECISION ===")
        print(json.dumps(result["decision"], indent=2))
       
        return
    elif args.command == "recognize":
     cfg = load_config(args.config)

     # Load screenshot
     import cv2
     img = cv2.imread(args.image)
     if img is None:
         raise SystemExit(f"Error: cannot read image {args.image}")

     # Extract UI elements (this produces sprite crops on disk)
     parse_result = extract_from_screenshot(img)
 
     # Recognize Pokémon
     sprites = parse_result.get("sprites", {})
     own_path = sprites.get("own_sprite_file")
     opp_path = sprites.get("opp_sprite_file")
 
     if not own_path or not opp_path:
         raise SystemExit("Sprite filenames missing in parse result.")
 
     our_name, our_conf = predict_image(own_path, cfg)
     opp_name, opp_conf = predict_image(opp_path, cfg)
 
     print("\n=== RECOGNITION ===")
     print(f"Our Pokémon     : {our_name}  (conf={our_conf:.3f})")
     print(f"Opponent Pokémon: {opp_name} (conf={opp_conf:.3f})")
 
     print("\nSprite crops:")
     print("  own:", own_path)
     print("  opp:", opp_path)
 
     return


if __name__ == "__main__":
    main()

