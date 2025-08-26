#!/usr/bin/env python3
"""
Test model forward pass
"""

import torch
import numpy as np
from HIST2ST import Hist2ST
from graph_construction import calcADJ

# 创建模型
model = Hist2ST(
    depth1=2, depth2=8, depth3=4,
    n_genes=785,
    kernel_size=5, patch_size=7,
    heads=16, channel=32, dropout=0.2,
    zinb=0.25, nb=False, bake=5, lamb=0.5
)

# 加载预训练权重
ckpt = torch.load('./model/5-Hist2ST.ckpt', map_location="cpu")
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    sd = ckpt['state_dict']
else:
    sd = ckpt

missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")

model.eval()

# 创建测试输入
batch_size = 1
n_spots = 10
patch_size = 112

# 创建patches
patches = torch.randn(batch_size, n_spots, 3, patch_size, patch_size)
print(f"Patches shape: {patches.shape}")

# 创建坐标
coords = torch.randint(0, 64, (batch_size, n_spots, 2))
print(f"Coords shape: {coords.shape}")

# 创建邻接矩阵
coords_np = coords.squeeze(0).numpy()
A = calcADJ(coords_np, k=6, pruneTag='Grid')
print(f"Adjacency matrix shape: {A.shape}")

# 前向传播
with torch.no_grad():
    try:
        out, _, _ = model(patches, coords, A)
        print(f"Output shape: {out.shape}")
        print(f"Output stats:")
        print(f"  - Min: {out.min().item():.6f}")
        print(f"  - Max: {out.max().item():.6f}")
        print(f"  - Mean: {out.mean().item():.6f}")
        print(f"  - Std: {out.std().item():.6f}")
        print(f"  - NaN count: {torch.isnan(out).sum().item()}")
        
        if torch.isnan(out).any():
            print("❌ Model output contains NaN values!")
        else:
            print("✅ Model output is valid!")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
