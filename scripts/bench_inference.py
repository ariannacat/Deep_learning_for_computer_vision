#!/usr/bin/env python3
"""
Benchmark model inference speed for sanity checks.
"""

import argparse
import time
from pokeai.vision import load_model, predict_species
from pokeai.config import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = load_model(cfg)

    t0 = time.time()
    label, conf = predict_species(args.image, model, cfg)
    dt = time.time() - t0

    print(f"Prediction: {label} ({conf:.3f}) in {dt:.4f}s")


if __name__ == "__main__":
    main()

