#!/bin/bash

# Hist2ST Analysis Script
# Usage: ./run_analysis.sh <SAMPLE_ID>

set -e  # Exit on any error

# Check if sample ID is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Sample ID is required"
    echo "Usage: ./run_analysis.sh <SAMPLE_ID>"
    echo "Example: ./run_analysis.sh MEND159"
    exit 1
fi

SAMPLE_ID=$1

echo "🚀 Starting spatial transcriptomics analysis for sample: $SAMPLE_ID"
echo "=================================================================="

# Check if prediction results exist
OUTPUT_DIR="output/$SAMPLE_ID"
PRED_FILE="$OUTPUT_DIR/predictions/${SAMPLE_ID}_pred.h5ad"

echo "📋 Checking prediction results..."

if [ ! -f "$PRED_FILE" ]; then
    echo "❌ Error: Prediction results not found: $PRED_FILE"
    echo "💡 Please run prediction first: ./run_prediction.sh $SAMPLE_ID"
    exit 1
fi

echo "✅ Prediction results found"

# Check if Python script exists
SCRIPT_FILE="analyze_hest_universal.py"
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "❌ Error: Analysis script not found: $SCRIPT_FILE"
    exit 1
fi

# Run analysis
echo "🔬 Running spatial transcriptomics analysis..."
python "$SCRIPT_FILE" "$SAMPLE_ID"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Analysis completed successfully!"
    echo "📊 Results saved in: $OUTPUT_DIR"
    echo ""
    echo "📋 Generated files:"
    echo "   - $OUTPUT_DIR/analysis/${SAMPLE_ID}_analyzed.h5ad"
    echo "   - $OUTPUT_DIR/analysis/clustering_results.csv"
    echo "   - $OUTPUT_DIR/analysis/marker_genes.csv"
    echo "   - $OUTPUT_DIR/visualizations/ (generated plots)"
    echo "   - $OUTPUT_DIR/logs/ (processing logs)"
    echo ""
    echo "📈 Analysis includes:"
    echo "   - Quality control and filtering"
    echo "   - Normalization and scaling"
    echo "   - Dimensionality reduction (PCA, UMAP, t-SNE)"
    echo "   - Clustering analysis (Leiden, Louvain, K-means)"
    echo "   - Differential expression analysis"
    echo "   - Spatial visualization"
    echo "   - Quality assessment"
    echo ""
    echo "🎨 Visualizations available in: $OUTPUT_DIR/visualizations/"
else
    echo "❌ Analysis failed!"
    exit 1
fi
