#!/bin/bash
# Usage:
#   chmod +x scripts/download_models.sh    # make executable (run once)
#   ./scripts/download_models.sh           # run script
#
# This script downloads model weights from the artifacts folder

#!/bin/bash
# scripts/download_models.sh
# Now: collect models from artifacts/ instead of downloading.

set -e

echo "Collecting model weights from artifacts/..."

mkdir -p models/

# Copy all Torch / YOLO weights from artifacts into models/
find artifacts -type f \( -name "*.pt" -o -name "*.pth" \) -exec cp {} models/ \;

echo "Done. Copied weights to models/."

