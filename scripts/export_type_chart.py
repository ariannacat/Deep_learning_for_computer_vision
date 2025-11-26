#!/usr/bin/env python3
"""
Export the internal type chart dictionary to JSON (optional).
"""

import json
from pokeai.type_chart import type_chart

def main():
    out_path = "data/type_charts/type_chart.json"
    with open(out_path, "w") as f:
        json.dump(type_chart, f, indent=2)
    print(f"Exported type chart to {out_path}")

if __name__ == "__main__":
    main()
