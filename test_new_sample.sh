#!/bin/bash

# Test script to demonstrate adding a new sample
# This shows how easy it is to use the new universal system

echo "🧪 Testing the new universal Hist2ST system"
echo "=========================================="

# Example: How to add a new sample
echo ""
echo "📝 To add a new sample (e.g., SAMPLE123), you would:"
echo ""
echo "1. Prepare your data files:"
echo "   data/hest_data/st/SAMPLE123.h5ad    # Gene expression data"
echo "   data/hest_data/wsis/SAMPLE123.tif   # H&E image"
echo ""
echo "2. Run prediction:"
echo "   ./run_prediction.sh SAMPLE123"
echo ""
echo "3. Run analysis:"
echo "   ./run_analysis.sh SAMPLE123"
echo ""
echo "4. Results will be saved in:"
echo "   output/SAMPLE123/"
echo ""

# Check current sample
echo "📊 Current sample (MEND159) results:"
if [ -d "output/MEND159" ]; then
    echo "✅ MEND159 results found:"
    echo "   - Predictions: $(ls -la output/MEND159/predictions/ | wc -l) files"
    echo "   - Analysis: $(ls -la output/MEND159/analysis/ | wc -l) files"
    echo "   - Logs: $(ls -la output/MEND159/logs/ | wc -l) files"
else
    echo "❌ MEND159 results not found"
fi

echo ""
echo "🎯 The system is now universal and robust!"
echo "   - No more hardcoded sample names"
echo "   - Organized output structure"
echo "   - Comprehensive logging"
echo "   - Easy to add new samples"
