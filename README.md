# Hist2ST: Spatial Transcriptomics Prediction from Histology

Transformer + GNN model to predict spatial gene expression from H&E histology.
Original project authors: Yuansong Zeng, Zhuoyi Wei, Weijiang Yu, Rui Yin, Bingling Li, Zhonghui Tang, Yutong Lu, Yuedong Yang.

This repository adds a robust, universal pipeline to run zero-shot inference on external HEST data and a complete downstream analysis workflow.

## Overview

- **Model**: Hist2ST combines CNN/Transformer for global context and GNN for local spatial structure; predicts gene expression (ZINB/NB heads).
- **Our contribution**: A universal, sample-agnostic inference + analysis pipeline for HEST data with clean outputs and shell wrappers.
- **Upstream usage**: Training/tutorial notebooks from the original repo are kept for reference.

## Quick Start (External HEST Inference)

```bash
# 1) Ensure data + model are present
#    data/hest_data/st/{SAMPLE_ID}.h5ad
#    data/hest_data/wsis/{SAMPLE_ID}.tif
#    model/5-Hist2ST.ckpt

# 2) Make scripts executable (first time only)
chmod +x run_prediction.sh run_analysis.sh

# 3) Run prediction and analysis
./run_prediction.sh MEND159
./run_analysis.sh MEND159
```

## Input layout

```
data/hest_data/
├── st/{SAMPLE_ID}.h5ad          # counts + spatial
└── wsis/{SAMPLE_ID}.tif         # H&E fallback image

model/
└── 5-Hist2ST.ckpt               # pretrained weights
```

Optional gene list: `data/her_hvg_cut_1000.npy` (first 785 used if present).

## Output layout

```
output/{SAMPLE_ID}/
├── predictions/
│   ├── {SAMPLE_ID}_pred.h5ad         # 785-gene predictions
│   └── correlation_results.npy       # Pearson/Spearman + overlap genes
├── analysis/
│   ├── {SAMPLE_ID}_analyzed.h5ad     # processed AnnData
│   ├── clustering_results.csv
│   └── marker_genes.csv
├── visualizations/                   # UMAP/t-SNE/spatial plots
└── logs/                             # pipeline logs
```

## Commands and scripts

- `run_prediction.sh SAMPLE_ID` — shell wrapper for inference
- `run_analysis.sh SAMPLE_ID` — shell wrapper for downstream analysis
- `predict_hest_universal.py` — universal prediction (loads .h5ad + .tif, builds KNN graph, runs Hist2ST)
- `analyze_hest_universal.py` — QC, HVG, PCA/UMAP/t-SNE, clustering, DE, spatial plots

Advanced (Python flags):
```bash
python predict_hest_universal.py SAMPLE_ID \
  --device auto --data_dir data/hest_data --output_dir output
```

## Technical notes

- Config: `5-7-2-8-4-16-32`, `n_genes=785`, dropout=0.2
- Weights: loaded with `strict=False` to allow partial compatibility
- Graph: `k=6`; dynamically switches `pruneTag` (Grid/NA) by coordinate range
- Coordinates: normalized to integer indices (0–63) for embeddings
- Seeds fixed (12000) for reproducibility

## Minimal model usage (reference)

```python
import torch
from HIST2ST import Hist2ST

model = Hist2ST(depth1=2, depth2=8, depth3=4,
                n_genes=785, kernel_size=5, patch_size=7,
                heads=16, channel=32, dropout=0.2,
                zinb=0.25, nb=False, bake=5, lamb=0.5)
# patches: [B, N, 3, H, W]
# coords:  [B, N, 2] (long indices 0..63)
# adj:     [N, N]
# out:     [B, N, n_genes]
```

## Requirements

- Python >= 3.7, PyTorch >= 1.10, pytorch-lightning >= 1.4, scanpy >= 1.8, scipy, PIL, tqdm

## Troubleshooting

- "Pre-trained model not found": put `5-Hist2ST.ckpt` under `model/`
- "No overlapping genes": confirm `.npy` gene list or remove it to use dataset genes
- Very low correlations: expected in zero-shot cross-dataset; predictions can still be useful
- PIL DecompressionBombWarning: safe for large WSIs

## Datasets (upstream)

- HER2+ breast tumor ST: `https://github.com/almaan/her2st`
- cSCC 10x Visium (GSE144240)
- Synapse mirror of trained models and data indices (see upstream paper)

## Citation (upstream)

Please cite the original authors:
```
@article{zengys,
  title={Spatial Transcriptomics Prediction from Histology jointly through Transformer and Graph Neural Networks},
  author={Yuansong Zeng and Zhuoyi Wei and Weijiang Yu and Rui Yin and Bingling Li and Zhonghui Tang and Yutong Lu and Yuedong Yang},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
