#!/usr/bin/env python3
"""
Universal spatial transcriptomics analysis script for HEST samples
Usage: python analyze_hest_universal.py <SAMPLE_ID>
"""

import os
import sys
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings
import argparse
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# Set scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

def setup_logging(sample_id, output_dir):
    """设置日志"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{sample_id}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_and_preprocess_data(sample_id, config):
    """加载和预处理数据"""
    logger = logging.getLogger(__name__)
    
    # 加载预测结果
    pred_file = os.path.join(config['output_dir'], sample_id, "predictions", f"{sample_id}_pred.h5ad")
    if not os.path.isfile(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    
    adata_pred = sc.read_h5ad(pred_file)
    logger.info(f"Loaded prediction data: {adata_pred.shape}")
    
    # 加载原始数据（用于对比）
    original_file = os.path.join(config['data_dir'], 'st', f'{sample_id}.h5ad')
    adata_orig = None
    if os.path.isfile(original_file):
        adata_orig = sc.read_h5ad(original_file)
        logger.info(f"Loaded original data: {adata_orig.shape}")
    else:
        logger.warning(f"Original data not found: {original_file}")
    
    return adata_pred, adata_orig

def basic_qc_and_filtering(adata):
    """基本质量控制和过滤"""
    logger = logging.getLogger(__name__)
    
    # 检查数据是否为空或全为0
    if adata.X.size == 0:
        logger.warning("Data is empty!")
        return adata
    
    # 检查是否有NaN值
    if hasattr(adata.X, 'toarray'):
        X_array = adata.X.toarray()
    else:
        X_array = adata.X
    
    nan_count = np.isnan(X_array).sum()
    zero_count = (X_array == 0).sum()
    total_count = X_array.size
    
    logger.info(f"Data quality check:")
    logger.info(f"  - Total values: {total_count}")
    logger.info(f"  - NaN values: {nan_count} ({nan_count/total_count*100:.2f}%)")
    logger.info(f"  - Zero values: {zero_count} ({zero_count/total_count*100:.2f}%)")
    logger.info(f"  - Non-zero values: {total_count - zero_count - nan_count}")
    
    # 如果数据全为0或NaN，跳过过滤
    if zero_count + nan_count == total_count:
        logger.warning("All values are 0 or NaN - skipping filtering")
        return adata
    
    # 计算基本统计
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # 过滤低质量spots和基因（只有在数据有效时）
    if adata.n_obs > 0 and adata.n_vars > 0:
        try:
            sc.pp.filter_cells(adata, min_genes=1)  # 降低阈值
            sc.pp.filter_genes(adata, min_cells=1)  # 降低阈值
            logger.info(f"After filtering: {adata.shape}")
        except:
            logger.warning("Filtering failed, keeping original data")
    
    return adata

def normalize_and_scale(adata):
    """标准化和缩放"""
    logger = logging.getLogger(__name__)
    
    # 检查数据是否有效
    if adata.X.size == 0:
        logger.warning("Data is empty - skipping normalization")
        return adata
    
    if hasattr(adata.X, 'toarray'):
        X_array = adata.X.toarray()
    else:
        X_array = adata.X
    
    # 检查是否全为0或NaN
    if np.all(X_array == 0) or np.all(np.isnan(X_array)):
        logger.warning("All values are 0 or NaN - skipping normalization")
        return adata
    
    try:
        # 标准化到10,000 reads per cell
        sc.pp.normalize_total(adata, target_sum=1e4)
        
        # Log transform
        sc.pp.log1p(adata)
        
        # 识别高变异基因（降低阈值）
        sc.pp.highly_variable_genes(adata, min_mean=0.001, max_mean=10, min_disp=0.1)
        logger.info(f"Highly variable genes: {sum(adata.var.highly_variable)}")
        
        # 缩放到单位方差
        sc.pp.scale(adata, max_value=10)
        
    except Exception as e:
        logger.warning(f"Normalization failed: {e}")
        logger.info("Proceeding with raw data")
    
    return adata

def dimensionality_reduction(adata):
    """降维分析"""
    logger = logging.getLogger(__name__)
    
    # 检查数据是否有效
    if adata.X.size == 0:
        logger.warning("Data is empty - skipping dimensionality reduction")
        return adata
    
    try:
        # PCA
        sc.tl.pca(adata, use_highly_variable=True, svd_solver='arpack')
        
        # 计算邻居图
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        
        # UMAP
        sc.tl.umap(adata)
        
        # t-SNE
        sc.tl.tsne(adata, use_rep='X_pca')
        
        logger.info(f"PCA components: {adata.obsm['X_pca'].shape}")
        
    except Exception as e:
        logger.warning(f"Dimensionality reduction failed: {e}")
    
    return adata

def clustering_analysis(adata):
    """聚类分析"""
    logger = logging.getLogger(__name__)
    
    # 检查是否有PCA结果
    if 'X_pca' not in adata.obsm:
        logger.warning("No PCA results - skipping clustering")
        return adata
    
    try:
        # Leiden聚类
        sc.tl.leiden(adata, resolution=0.5)
        
        # Louvain聚类
        sc.tl.louvain(adata, resolution=0.5)
        
        # K-means聚类（基于PCA）
        pca_data = adata.obsm['X_pca'][:, :20]
        kmeans = KMeans(n_clusters=8, random_state=42)
        adata.obs['kmeans_clusters'] = kmeans.fit_predict(pca_data)
        
        logger.info(f"Clustering results:")
        logger.info(f"  - Leiden clusters: {adata.obs['leiden'].nunique()}")
        logger.info(f"  - Louvain clusters: {adata.obs['louvain'].nunique()}")
        logger.info(f"  - K-means clusters: {adata.obs['kmeans_clusters'].nunique()}")
        
    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
    
    return adata

def differential_expression_analysis(adata):
    """差异表达分析"""
    logger = logging.getLogger(__name__)
    
    # 检查是否有聚类结果
    if 'leiden' not in adata.obs.columns:
        logger.warning("No clustering results - skipping DE analysis")
        return adata
    
    try:
        # 使用Leiden聚类进行差异表达分析
        sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
        
        # 获取每个cluster的marker基因
        result = adata.uns['rank_genes_groups']
        groups = result['names'].dtype.names
        
        logger.info(f"DE analysis completed for {len(groups)} clusters")
        
        # 显示每个cluster的top marker基因
        for group in groups[:5]:  # 只显示前5个clusters
            logger.info(f"Cluster {group} top markers:")
            for i in range(min(5, len(result['names'][group]))):
                gene = result['names'][group][i]
                score = result['scores'][group][i]
                logger.info(f"  {gene}: {score:.3f}")
        
    except Exception as e:
        logger.warning(f"DE analysis failed: {e}")
    
    return adata

def visualization_plots(adata, adata_orig, sample_id, output_dir):
    """可视化图表"""
    logger = logging.getLogger(__name__)
    
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        # 设置图形大小
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. UMAP聚类图
        if 'X_umap' in adata.obsm and 'leiden' in adata.obs.columns:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # UMAP with different clustering methods
            sc.pl.umap(adata, color='leiden', ax=axes[0,0], show=False, title='Leiden Clustering')
            sc.pl.umap(adata, color='louvain', ax=axes[0,1], show=False, title='Louvain Clustering')
            sc.pl.umap(adata, color='kmeans_clusters', ax=axes[0,2], show=False, title='K-means Clustering')
            
            # t-SNE plots
            sc.pl.tsne(adata, color='leiden', ax=axes[1,0], show=False, title='t-SNE Leiden')
            sc.pl.tsne(adata, color='louvain', ax=axes[1,1], show=False, title='t-SNE Louvain')
            sc.pl.tsne(adata, color='kmeans_clusters', ax=axes[1,2], show=False, title='t-SNE K-means')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'clustering_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved clustering analysis plots")
        
        # 2. 空间聚类图
        if 'spatial' in adata.obsm and 'leiden' in adata.obs.columns:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            sc.pl.spatial(adata, color='leiden', ax=axes[0], show=False, title='Spatial Leiden')
            sc.pl.spatial(adata, color='louvain', ax=axes[1], show=False, title='Spatial Louvain')
            sc.pl.spatial(adata, color='kmeans_clusters', ax=axes[2], show=False, title='Spatial K-means')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'spatial_clustering.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved spatial clustering plots")
        
        # 3. 差异表达热图
        if 'rank_genes_groups' in adata.uns:
            sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, use_raw=False, 
                                           save=f'_{sample_id}_marker_genes_heatmap.png')
            logger.info("Saved marker genes heatmap")
        
        # 4. 基因表达空间图（选择几个marker基因）
        if 'spatial' in adata.obsm and 'rank_genes_groups' in adata.uns:
            # 获取top marker基因
            top_genes = []
            for group in adata.uns['rank_genes_groups']['names'].dtype.names[:3]:
                top_genes.extend(adata.uns['rank_genes_groups']['names'][group][:3])
            
            # 绘制空间表达图
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.flatten()
            
            for i, gene in enumerate(top_genes[:9]):
                if gene in adata.var_names:
                    sc.pl.spatial(adata, color=gene, ax=axes[i], show=False, 
                                 title=f'{gene} Expression', use_raw=False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'spatial_gene_expression.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved spatial gene expression plots")
        
        # 5. 与原始数据对比（如果有）
        if adata_orig is not None:
            try:
                # 找到共同的基因
                common_genes = set(adata.var_names) & set(adata_orig.var_names)
                if len(common_genes) > 0:
                    common_genes = list(common_genes)[:5]  # 选择前5个基因
                    
                    fig, axes = plt.subplots(2, len(common_genes), figsize=(4*len(common_genes), 8))
                    
                    for i, gene in enumerate(common_genes):
                        # 预测结果
                        sc.pl.spatial(adata, color=gene, ax=axes[0, i], show=False, 
                                     title=f'Predicted {gene}', use_raw=False)
                        # 原始数据
                        sc.pl.spatial(adata_orig, color=gene, ax=axes[1, i], show=False, 
                                     title=f'Original {gene}', use_raw=False)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'prediction_vs_original.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("Saved prediction vs original comparison plots")
            except:
                logger.warning("Could not create comparison plots")
        
        logger.info("All visualization plots saved")
        
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

def quality_assessment(adata, adata_orig, sample_id):
    """质量评估"""
    logger = logging.getLogger(__name__)
    
    # 1. 表达分布
    logger.info("Expression distribution:")
    if hasattr(adata.X, 'toarray'):
        X_array = adata.X.toarray()
    else:
        X_array = adata.X
    
    logger.info(f"  - Mean expression: {np.mean(X_array):.4f}")
    logger.info(f"  - Std expression: {np.std(X_array):.4f}")
    logger.info(f"  - Zero rate: {(X_array == 0).sum() / X_array.size:.2%}")
    
    # 2. 聚类质量
    logger.info("Clustering quality:")
    for method in ['leiden', 'louvain', 'kmeans_clusters']:
        if method in adata.obs.columns:
            n_clusters = adata.obs[method].nunique()
            logger.info(f"  - {method}: {n_clusters} clusters")
    
    # 3. 与原始数据对比（如果有）
    if adata_orig is not None:
        try:
            # 找到共同基因
            common_genes = set(adata.var_names) & set(adata_orig.var_names)
            if len(common_genes) > 0:
                common_genes = list(common_genes)[:100]  # 选择前100个基因
                
                # 计算相关性
                pred_idx = [list(adata.var_names).index(g) for g in common_genes]
                orig_idx = [list(adata_orig.var_names).index(g) for g in common_genes]
                
                correlations = []
                for p_idx, o_idx in zip(pred_idx, orig_idx):
                    try:
                        corr = pearsonr(adata.X[:, p_idx], adata_orig.X[:, o_idx])[0]
                        correlations.append(corr)
                    except:
                        correlations.append(np.nan)
                
                correlations = [c for c in correlations if not np.isnan(c)]
                if correlations:
                    logger.info(f"Prediction vs Original correlation:")
                    logger.info(f"  - Mean Pearson r: {np.mean(correlations):.4f}")
                    logger.info(f"  - Median Pearson r: {np.median(correlations):.4f}")
                    logger.info(f"  - Correlation range: [{np.min(correlations):.4f}, {np.max(correlations):.4f}]")
        except:
            logger.warning("Could not calculate prediction vs original correlation")
    
    return adata

def save_analysis_results(adata, sample_id, output_dir):
    """保存分析结果"""
    logger = logging.getLogger(__name__)
    
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    try:
        # 保存处理后的数据
        processed_file = os.path.join(analysis_dir, f"{sample_id}_analyzed.h5ad")
        adata.write_h5ad(processed_file)
        logger.info(f"Saved processed data to: {processed_file}")
        
        # 保存聚类结果
        cluster_columns = [col for col in ['leiden', 'louvain', 'kmeans_clusters'] if col in adata.obs.columns]
        if cluster_columns:
            cluster_results = adata.obs[cluster_columns].copy()
            cluster_file = os.path.join(analysis_dir, "clustering_results.csv")
            cluster_results.to_csv(cluster_file)
            logger.info(f"Saved clustering results to: {cluster_file}")
        
        # 保存marker基因
        if 'rank_genes_groups' in adata.uns:
            result = adata.uns['rank_genes_groups']
            groups = result['names'].dtype.names
            
            marker_genes = {}
            for group in groups:
                genes = result['names'][group][:20]  # Top 20 genes
                scores = result['scores'][group][:20]
                marker_genes[group] = list(zip(genes, scores))
            
            # 保存为CSV
            max_genes = max(len(genes) for genes in marker_genes.values())
            df_markers = pd.DataFrame(index=range(max_genes))
            
            for group, genes in marker_genes.items():
                gene_names = [g[0] for g in genes]
                gene_scores = [g[1] for g in genes]
                
                # 填充到相同长度
                while len(gene_names) < max_genes:
                    gene_names.append('')
                    gene_scores.append(np.nan)
                
                df_markers[f'{group}_gene'] = gene_names
                df_markers[f'{group}_score'] = gene_scores
            
            marker_file = os.path.join(analysis_dir, "marker_genes.csv")
            df_markers.to_csv(marker_file, index=False)
            logger.info(f"Saved marker genes to: {marker_file}")
        
        logger.info("All analysis results saved")
        
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")

def analyze_sample(sample_id, config):
    """对单个样本进行分析"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting analysis for sample: {sample_id}")
    
    # 检查输出目录
    output_dir = os.path.join(config['output_dir'], sample_id)
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # 设置日志
    logger = setup_logging(sample_id, output_dir)
    
    # 1. 加载数据
    adata_pred, adata_orig = load_and_preprocess_data(sample_id, config)
    
    # 2. 基本QC和过滤
    adata_pred = basic_qc_and_filtering(adata_pred)
    
    # 3. 标准化和缩放
    adata_pred = normalize_and_scale(adata_pred)
    
    # 4. 降维分析
    adata_pred = dimensionality_reduction(adata_pred)
    
    # 5. 聚类分析
    adata_pred = clustering_analysis(adata_pred)
    
    # 6. 差异表达分析
    adata_pred = differential_expression_analysis(adata_pred)
    
    # 7. 可视化
    visualization_plots(adata_pred, adata_orig, sample_id, output_dir)
    
    # 8. 质量评估
    quality_assessment(adata_pred, adata_orig, sample_id)
    
    # 9. 保存结果
    save_analysis_results(adata_pred, sample_id, output_dir)
    
    logger.info(f"Analysis completed successfully for {sample_id}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Spatial transcriptomics analysis for HEST samples')
    parser.add_argument('sample_id', type=str, help='Sample ID (e.g., MEND159)')
    parser.add_argument('--data_dir', type=str, default='data/hest_data', 
                       help='Base directory for HEST data')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Base directory for output')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir
    }
    
    try:
        output_dir = analyze_sample(args.sample_id, config)
        print(f"✅ Analysis completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
