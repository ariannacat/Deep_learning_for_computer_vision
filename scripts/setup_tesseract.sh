#!/bin/bash

# ------------------------------------------------------------
# setup_tesseract.sh
#
# Usage:
#   chmod +x scripts/setup_tesseract.sh   # make script executable (run once)
#   ./scripts/setup_tesseract.sh          # run the installer
#
# Purpose:
#   Installs the Tesseract OCR engine required by pytesseract.
#   Different OS use different install commands.
# ------------------------------------------------------------

echo "Installing Tesseract OCR..."

OS=$(uname)

if [[ "$OS" == "Darwin" ]]; then
    brew install tesseract
elif [[ "$OS" == "Linux" ]]; then
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr
else
    echo "Please install Tesseract manually for OS: $OS"
fi

