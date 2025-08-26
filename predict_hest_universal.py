#!/usr/bin/env python3
"""
Universal Hist2ST prediction script for HEST samples
Usage: python predict_hest_universal.py <SAMPLE_ID>
"""

import os
import sys
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import random
import argparse
from datetime import datetime
import logging

# Import Hist2ST model and utilities
try:
    from HIST2ST import Hist2ST
    from graph_construction import calcADJ
except ImportError:
    print("Warning: HIST2ST not found, using placeholder")
    class Hist2ST:
        def __init__(self, **kwargs):
            pass
        def eval(self):
            return self
        def to(self, device):
            return self

def setup_logging(sample_id, output_dir):
    """设置日志"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{sample_id}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def create_output_structure(sample_id, base_output_dir="output"):
    """创建输出目录结构"""
    output_dir = os.path.join(base_output_dir, sample_id)
    
    # 创建目录结构
    dirs = [
        output_dir,
        os.path.join(output_dir, "predictions"),
        os.path.join(output_dir, "analysis"),
        os.path.join(output_dir, "visualizations"),
        os.path.join(output_dir, "logs")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return output_dir

def load_img_and_scale(adata, fallback_path):
    """加载和缩放图像"""
    img, scale = None, None
    
    # 尝试从AnnData中加载图像
    if "spatial" in adata.uns and len(adata.uns["spatial"]) > 0:
        lib = list(adata.uns["spatial"].keys())[0]
        try:
            hires = adata.uns["spatial"][lib]["images"].get("hires", None)
            if isinstance(hires, np.ndarray):
                arr = (hires*255).astype(np.uint8) if hires.max() <= 1 else hires.astype(np.uint8)
                img = Image.fromarray(arr).convert("RGB")
            elif isinstance(hires, (str, bytes)):
                img = Image.open(hires).convert("RGB")
            scale = adata.uns["spatial"][lib]["scalefactors"].get("tissue_hires_scalef", None)
        except Exception as e:
            logging.warning(f"Failed to load image from AnnData: {e}")
    
    # 如果AnnData中没有图像，使用fallback路径
    if img is None and os.path.isfile(fallback_path):
        try:
            img = Image.open(fallback_path).convert("RGB")
            logging.info(f"Loaded image from fallback path: {fallback_path}")
        except Exception as e:
            logging.error(f"Failed to load image from fallback path: {e}")
    
    return img, scale

def crop(img, xy, size):
    """裁剪图像patch"""
    x, y = float(xy[0]), float(xy[1])
    half = size // 2
    W, H = img.size
    l, u = int(round(x-half)), int(round(y-half))
    r, b = l + size, u + size
    pl, pt, pr, pb = max(0, -l), max(0, -u), max(0, r-W), max(0, b-H)
    l, u = max(0, l), max(0, u)
    r, b = min(W, r), min(H, b)
    patch = np.array(img.crop((l, u, r, b)))
    if any([pl, pt, pr, pb]):
        patch = np.pad(patch, ((pt, pb), (pl, pr), (0, 0)), mode="constant", constant_values=0)
    return torch.from_numpy(patch).permute(2, 0, 1).float() / 255.

def normalize_coords(coords, n_pos=64):
    """归一化坐标"""
    coords = coords.astype(np.float32)
    
    # 检查坐标范围
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    
    logging.info(f"Coordinate ranges: x={x_range:.2f}, y={y_range:.2f}")
    
    # 如果坐标范围太小，使用默认值
    if x_range < 1e-6 or y_range < 1e-6:
        logging.warning("Coordinates have very small range, using default values")
        coords_normalized = np.zeros((len(coords), 2), dtype=np.int64)
        for i in range(len(coords)):
            coords_normalized[i, 0] = i % n_pos
            coords_normalized[i, 1] = (i // n_pos) % n_pos
        return coords_normalized
    
    # 归一化到[0, n_pos-1]
    coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / x_range * (n_pos - 1)
    coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / y_range * (n_pos - 1)
    return coords.astype(np.int64)

def predict_sample(sample_id, config):
    """对单个样本进行预测"""
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = create_output_structure(sample_id, config['base_output_dir'])
    
    # 设置日志
    logger = setup_logging(sample_id, output_dir)
    
    logger.info(f"Starting prediction for sample: {sample_id}")
    
    # 检查输入文件
    h5ad_path = os.path.join(config['data_dir'], 'st', f'{sample_id}.h5ad')
    img_path = os.path.join(config['data_dir'], 'wsis', f'{sample_id}.tif')
    
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(f"Gene expression file not found: {h5ad_path}")
    
    # 加载数据
    logger.info(f"Loading data from {h5ad_path}")
    ad = sc.read_h5ad(h5ad_path)
    logger.info(f"Data shape: {ad.shape}")
    
    # 加载图像
    img, scale = load_img_and_scale(ad, img_path)
    if img is None:
        raise ValueError(f"No H&E image found for sample {sample_id}")
    logger.info(f"Image loaded: {img.size}")
    
    # 读取基因列表
    expected_path = "data/her_hvg_cut_1000.npy"
    if os.path.isfile(expected_path):
        expected = list(np.load(expected_path, allow_pickle=True))
        expected = expected[:config['n_genes']]
        logger.info(f"Using expected genes: {len(expected)}")
    else:
        expected = list(ad.var_names)
        logger.info(f"Using all genes from data: {len(expected)}")
    
    # 计算基因交集
    var_to_idx = {g: i for i, g in enumerate(ad.var_names)}
    overlap_idx = [var_to_idx[g] for g in expected if g in var_to_idx]
    overlap_genes = [g for g in expected if g in var_to_idx]
    n_genes = len(overlap_genes)
    
    if n_genes == 0:
        raise ValueError(f"No overlapping genes between model and {sample_id}")
    
    logger.info(f"Overlap genes: {n_genes}")
    
    # 获取坐标和基因表达数据
    coords = np.array(ad.obsm["spatial"], dtype=float)
    X = ad.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = X.astype(np.float32)[:, overlap_idx]
    
    logger.info(f"Processing {len(coords)} spots...")
    
    # 构建邻接矩阵
    coord_range = np.max(coords) - np.min(coords)
    logger.info(f"Coordinate range: {coord_range}")
    
    if coord_range > 100:
        A = calcADJ(coords, k=config['knn_k'], pruneTag='NA')
    else:
        A = calcADJ(coords, k=config['knn_k'], pruneTag='Grid')
    
    A = A.to(config['device'])
    logger.info(f"Adjacency matrix shape: {A.shape}")
    logger.info(f"Adjacency matrix connections: {A.sum().item()}")
    
    # 创建模型
    logger.info(f"Creating model with config: k={config['kernel_size']}, p={config['patch_size']}, "
                f"d1={config['depth1']}, d2={config['depth2']}, d3={config['depth3']}, "
                f"h={config['heads']}, c={config['channel']}")
    
    model = Hist2ST(
        depth1=config['depth1'], depth2=config['depth2'], depth3=config['depth3'],
        n_genes=config['n_genes'],
        kernel_size=config['kernel_size'], patch_size=config['patch_size'],
        heads=config['heads'], channel=config['channel'], dropout=config['dropout'],
        zinb=config['zinb'], nb=config['nb'], bake=config['bake'], lamb=config['lamb']
    ).to(config['device']).eval()
    
    # 加载预训练权重
    if os.path.isfile(config['checkpoint_path']):
        logger.info(f"Loading pre-trained weights from {config['checkpoint_path']}")
        ckpt = torch.load(config['checkpoint_path'], map_location="cpu")
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f"[ckpt] missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        raise FileNotFoundError(f"Pre-trained weights not found at {config['checkpoint_path']}")
    
    # 提取patches
    logger.info("Extracting patches for all spots...")
    patches = []
    
    # 归一化坐标
    logger.info("Normalizing coordinates...")
    normalized_coords = normalize_coords(coords, n_pos=64)
    
    for i in tqdm(range(len(coords)), desc="Extracting patches"):
        patch = crop(img, coords[i], config['fig_size'])
        patches.append(patch)
    
    # 转换为tensor
    patches = torch.stack(patches, dim=0)
    normalized_coords = torch.tensor(normalized_coords, dtype=torch.long)
    
    # 添加batch维度
    patches = patches.unsqueeze(0)
    normalized_coords = normalized_coords.unsqueeze(0)
    
    logger.info(f"Patches shape: {patches.shape}")
    logger.info(f"Coords shape: {normalized_coords.shape}")
    
    # 推理
    logger.info("Running inference...")
    with torch.no_grad():
        patches = patches.to(config['device'])
        normalized_coords = normalized_coords.to(config['device'])
        
        out, _, _ = model(patches, normalized_coords, A)
        preds = out.squeeze(0).detach().cpu().numpy()
    
    logger.info(f"Predictions shape: {preds.shape}")
    
    # 检查预测值
    logger.info(f"Prediction stats:")
    logger.info(f"  - Min: {preds.min():.6f}")
    logger.info(f"  - Max: {preds.max():.6f}")
    logger.info(f"  - Mean: {preds.mean():.6f}")
    logger.info(f"  - Std: {preds.std():.6f}")
    logger.info(f"  - NaN count: {np.isnan(preds).sum()}")
    
    # 评估相关性
    preds_overlap = preds[:, :n_genes]
    
    gene_r = []
    for g in range(n_genes):
        try:
            gene_r.append(pearsonr(preds_overlap[:, g], X[:, g])[0])
        except:
            gene_r.append(np.nan)
    
    spot_r = []
    for i in range(preds_overlap.shape[0]):
        try:
            spot_r.append(spearmanr(preds_overlap[i, :], X[i, :])[0])
        except:
            spot_r.append(np.nan)
    
    logger.info(f"[{sample_id}] overlap genes: {n_genes}")
    logger.info(f"[{sample_id}] median gene-wise Pearson: {np.nanmedian(gene_r):.4f}")
    logger.info(f"[{sample_id}] median spot-wise Spearman: {np.nanmedian(spot_r):.4f}")
    
    # 保存结果
    predictions_dir = os.path.join(output_dir, "predictions")
    
    # 保存预测结果
    ad_pred = sc.AnnData(preds)
    ad_pred.var_names = expected
    ad_pred.obs_names = ad.obs_names
    ad_pred.obsm["spatial"] = ad.obsm["spatial"]
    
    pred_file = os.path.join(predictions_dir, f"{sample_id}_pred.h5ad")
    ad_pred.write_h5ad(pred_file)
    logger.info(f"Saved predictions to: {pred_file}")
    
    # 保存相关性结果
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    corr_file = os.path.join(predictions_dir, f"{sample_id}_correlation_{ts}.npy")
    np.save(corr_file, {
        'gene_pearson': np.array(gene_r),
        'spot_spearman': np.array(spot_r),
        'overlap_genes': overlap_genes,
        'n_genes': n_genes
    })
    logger.info(f"Saved correlation results to: {corr_file}")
    
    logger.info(f"Prediction completed successfully for {sample_id}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Hist2ST prediction for HEST samples')
    parser.add_argument('sample_id', type=str, help='Sample ID (e.g., MEND159)')
    parser.add_argument('--data_dir', type=str, default='data/hest_data', 
                       help='Base directory for HEST data')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Base directory for output')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 模型配置
    config = {
        'data_dir': args.data_dir,
        'base_output_dir': args.output_dir,
        'device': device,
        'checkpoint_path': './model/5-Hist2ST.ckpt',
        'n_genes': 785,
        'fig_size': 112,
        'patch_size': 7,
        'kernel_size': 5,
        'depth1': 2,
        'depth2': 8,
        'depth3': 4,
        'heads': 16,
        'channel': 32,
        'dropout': 0.2,
        'zinb': 0.25,
        'nb': False,
        'bake': 5,
        'lamb': 0.5,
        'knn_k': 6
    }
    
    # 设置随机种子
    random.seed(12000)
    np.random.seed(12000)
    torch.manual_seed(12000)
    torch.cuda.manual_seed(12000)
    torch.cuda.manual_seed_all(12000)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    try:
        output_dir = predict_sample(args.sample_id, config)
        print(f"✅ Prediction completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
