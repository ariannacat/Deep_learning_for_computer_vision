#!/bin/bash
# Setup developer environment

# ------------------------------------------------------------
# Usage:
#   chmod +x scripts/dev_install.sh        # make script executable (run once)
#   ./scripts/dev_install.sh               # set up development environment
#
# Purpose:
#   Prepares a clean development environment for contributors.
#   - Upgrades pip
#   - Installs the project in editable mode with dev dependencies
#   - Installs pre-commit hooks
#
# Notes:
#   This script is for developers, not for end users.
#   It does NOT install Tesseract or model weights.
# ------------------------------------------------------------

python -m pip install -U pip
pip install -e ".[dev]"
pre-commit install

echo "Development environment ready!"

