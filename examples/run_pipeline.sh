#!/bin/bash
# Stop on first error
set -e

# Output file
OUTPUT_FILE="localize_data.js"
echo "const LOCALIZE_DATA = {};" > $OUTPUT_FILE

IMAGES=(
    "IndicDLP_6.png"
    "M6_1.png"
    "S1_raw.png"
    "SROIE_2.png"
)

# Use the syndoc environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate syndoc

for img in "${IMAGES[@]}"; do
    echo "==========================================="
    echo "Processing $img ..."
    echo "==========================================="
    python3 generate_localize_data.py \
        --image "$img" \
        --cfg ../configs/default.yaml \
        --output_file "$OUTPUT_FILE"
done

echo "All done! Results saved to $OUTPUT_FILE"
