#!/bin/bash

# Hist2ST Prediction Script
# Usage: ./run_prediction.sh <SAMPLE_ID>

set -e  # Exit on any error

# Check if sample ID is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Sample ID is required"
    echo "Usage: ./run_prediction.sh <SAMPLE_ID>"
    echo "Example: ./run_prediction.sh MEND159"
    exit 1
fi

SAMPLE_ID=$1

echo "🚀 Starting Hist2ST prediction for sample: $SAMPLE_ID"
echo "=================================================="

# Check if required files exist
DATA_DIR="data/hest_data"
H5AD_FILE="$DATA_DIR/st/${SAMPLE_ID}.h5ad"
IMG_FILE="$DATA_DIR/wsis/${SAMPLE_ID}.tif"
MODEL_FILE="./model/5-Hist2ST.ckpt"

echo "📋 Checking input files..."

if [ ! -f "$H5AD_FILE" ]; then
    echo "❌ Error: Gene expression file not found: $H5AD_FILE"
    exit 1
fi

if [ ! -f "$IMG_FILE" ]; then
    echo "❌ Error: H&E image file not found: $IMG_FILE"
    exit 1
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: Pre-trained model not found: $MODEL_FILE"
    exit 1
fi

echo "✅ All input files found"

# Check if Python script exists
SCRIPT_FILE="predict_hest_universal.py"
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "❌ Error: Prediction script not found: $SCRIPT_FILE"
    exit 1
fi

# Create output directory
OUTPUT_DIR="output/$SAMPLE_ID"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"

# Run prediction
echo "🔬 Running prediction..."
python "$SCRIPT_FILE" "$SAMPLE_ID"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Prediction completed successfully!"
    echo "📊 Results saved in: $OUTPUT_DIR"
    echo ""
    echo "📋 Generated files:"
    echo "   - $OUTPUT_DIR/predictions/${SAMPLE_ID}_pred.h5ad"
    echo "   - $OUTPUT_DIR/predictions/correlation_results.npy"
    echo "   - $OUTPUT_DIR/logs/ (processing logs)"
    echo ""
    echo "💡 Next step: Run analysis with: ./run_analysis.sh $SAMPLE_ID"
else
    echo "❌ Prediction failed!"
    exit 1
fi
