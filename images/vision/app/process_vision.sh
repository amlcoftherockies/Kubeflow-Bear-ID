#!/bin/bash
set -e

# Arguments
INPUT_RAW_DIR=""
OUTPUT_EMBEDDINGS_FILE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input-raw-dir) INPUT_RAW_DIR="$2"; shift ;;
        --output-embeddings-file) OUTPUT_EMBEDDINGS_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$INPUT_RAW_DIR" ] || [ -z "$OUTPUT_EMBEDDINGS_FILE" ]; then
    echo "Usage: $0 --input-raw-dir <dir> --output-embeddings-file <file>"
    exit 1
fi

echo "Processing images in $INPUT_RAW_DIR..."
# Ensure the output directory exists
mkdir -p "$(dirname "$OUTPUT_EMBEDDINGS_FILE")"

# Execute BearID Dlib binaries
# 1. imglab: Create initial XML from images
/opt/bearid-dlib/build/imglab -c /tmp/dataset.xml "$INPUT_RAW_DIR"

# 2. bearface: Detect faces
/opt/bearid-dlib/build/bearface --network /opt/bearid-models/bear_detector_mmod.dat --dataset /tmp/dataset.xml --output /tmp/faces.xml

# 3. bearchip: Extract and align face chips
/opt/bearid-dlib/build/bearchip --network /opt/bearid-models/bear_landmark_68_model.dat --dataset /tmp/faces.xml --output /tmp/chips.xml --chip-dir /tmp/chips

# 4. bearembed: Extract embeddings and output to CSV
/opt/bearid-dlib/build/bearembed --network /opt/bearid-models/bear_embed_network.dat --dataset /tmp/chips.xml --output-csv "$OUTPUT_EMBEDDINGS_FILE"

echo "Vision processing complete. Embeddings saved to $OUTPUT_EMBEDDINGS_FILE"
