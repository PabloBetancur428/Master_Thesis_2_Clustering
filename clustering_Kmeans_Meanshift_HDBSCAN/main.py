# main.py
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from data_loader import DataLoader
from feature_filter import FeatureFilter
from outlier_remover import OutlierRemover
from standardizer import Standardizer
from dimension_reducer import ClusterReducer
from clusterers import KMeansClusterer, MeanShiftClusterer, HDBSCANClusterer
from umap_visualizer import UMAPVisualizer
from visualizer import Visualizer
from lesion_filter import filter_by_voxel_count
import itertools


def run_clustering(
    data: pd.DataFrame,
    drop_cols: list = ['species'],
    # Voxel thresholds
    min_voxels: int = 50,
    max_voxels: int = None,
    # Outlier removal
    z_thresh: float = 3.0,
    # PCA
    n_pca: int = 4,
    # KMeans
    kmeans_k: int = 3,
    kmeans_init: str = 'k-means++',
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 300,
    # MeanShift
    meanshift_bw: float = None,
    meanshift_bin_seeding: bool = False,
    meanshift_cluster_all: bool = True,
    # HDBSCAN
    hdbscan_min_size: int = 5,
    hdbscan_min_samples: int = None,
    hdbscan_epsilon: float = 0.0,
    hdbscan_method: str = 'eom',
    # UMAP
    umap_components: int = 3,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = 'euclidean',
    # Visualization context
    title_prefix: str = '',
    random_state: int = 42
) -> tuple:
    """
    Executes the clustering pipeline with optional voxel filtering and
    appends a prefix to plot titles for context (e.g., modality name).

    Returns:
        tuple: (kmeans_labels, Meanshift_labels, HDBSCAN_labels)
    """
    # Step 1: optional filter by voxel count
    print(f"Before voxel filtering ({title_prefix}): {data.shape}")
    if 'num_voxels' in data.columns:
        data = filter_by_voxel_count(data, min_voxels, max_voxels)
    print(f"After voxel filtering ({title_prefix}): {data.shape}")

    # Step 2: extract ground truth
    truth = None
    if 'species' in data.columns:
        truth = pd.factorize(data['species'])[0]

    # Step 3: feature filtering
    features = FeatureFilter(drop_columns=drop_cols).filter(data)
    print(f"Features shape ({title_prefix}): {features.shape}")

    # Step 4: outlier removal
    #clean = OutlierRemover(z_thresh=z_thresh).remove_outliers(features)
    #print(f"After outlier removal ({title_prefix}): {clean.shape}")

    clean = features.copy()
    # Step 5: standardization
    scaled = Standardizer().fit_transform(clean)

    # Step 6: PCA reduction
    pca_data = ClusterReducer(n_components=n_pca, random_state=random_state).reduce(scaled)

    # Step 7: clustering models
    km_labels = KMeansClusterer(
        n_clusters=kmeans_k,
        init=kmeans_init,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
        random_state=random_state
    ).fit_predict(pca_data)
    ms_labels = MeanShiftClusterer(
        bandwidth=meanshift_bw,
        bin_seeding=meanshift_bin_seeding,
        cluster_all=meanshift_cluster_all
    ).fit_predict(pca_data)
    hb_labels = HDBSCANClusterer(
        min_cluster_size=hdbscan_min_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_epsilon=hdbscan_epsilon,
        cluster_selection_method=hdbscan_method
    ).fit_predict(pca_data)

    # Step 8: print cluster counts
    print("------------------------------------------")
    print(f"KMeans clusters ({title_prefix}): {len(np.unique(km_labels))}")
    print(f"MeanShift clusters ({title_prefix}): {len(np.unique(ms_labels))}")
    print(f"HDBSCAN clusters ({title_prefix}): {len(np.unique(hb_labels))}")

    # Step 9: UMAP visualization
    umap_data = UMAPVisualizer(
        n_components=umap_components,
        random_state=random_state,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric
    ).reduce(pca_data)

    # Step 10: plotting
    prefix = f"{title_prefix} - " if title_prefix else ''
    if truth is not None:
        Visualizer.plot_clusters(umap_data, truth, title=f"{prefix}Ground Truth")
    Visualizer.plot_clusters(umap_data, km_labels, title=f"{prefix}KMeans")
    Visualizer.plot_clusters(umap_data, ms_labels, title=f"{prefix}MeanShift")
    Visualizer.plot_clusters(umap_data, hb_labels, title=f"{prefix}HDBSCAN")

    return km_labels, ms_labels, hb_labels


def main():
    # Load dataset
    #data_path = "/home/jbetancur/Desktop/codes/clustering/data_iris_pruebas/archive/iris.csv"
    data_path = "/home/jbetancur/Desktop/codes/clustering/feature_extraction/output/Radiomic_QSM_Wavelet_LBP_2d_3d.csv"
    df = DataLoader(data_path).load()

    # Choose pipeline based on DataFrame columns
    if 'num_voxels' in df.columns:
        # Radiomics pipeline: 1) modality-dimension 2) base modalities 3) modality pairs
        full_mods = sorted({
            "_".join(c.split('_')[:2])
            for c in df.columns
            if '_' in c and c not in ['num_voxels','species','label_id','volume_physical']
        })
        bases = sorted({m.split('_')[0] for m in full_mods})
        dims = sorted({m.split('_')[1] for m in full_mods})
        
        print(f"Full modalities: {full_mods}")
        # 1) modality-dimension
        for mod in full_mods:
            print("\n")
            print(f"Modality: {mod}")
            if "2d" in mod:
                continue
            cols = [c for c in df.columns if c.startswith(f"{mod}_")]
            subset = cols #+ [c for c in ['num_voxels','species'] if c in df.columns]
            print(f"Subset: {subset}")
            print(f"\n=== Clustering: {mod} ===")
            run_clustering(
                df[subset].copy(),
                drop_cols=['volume_physical','species','num_voxels', 'QSM_2d_error'],
                min_voxels=12,
                max_voxels=10000,
                z_thresh=3.5,
                n_pca=None,
                kmeans_k=2,
                kmeans_init='random',
                kmeans_n_init=20,
                kmeans_max_iter=500,
                meanshift_bw=None,
                meanshift_bin_seeding=True,
                meanshift_cluster_all=False,
                hdbscan_min_size=15,
                hdbscan_min_samples=20,
                hdbscan_epsilon=0.2,
                hdbscan_method='eom',
                umap_components=3,
                umap_n_neighbors=30,
                umap_min_dist=0.2,
                umap_metric='cosine',
                title_prefix=mod,
                random_state=42
            )

        # 2) base modality (merge dims)
        for base in bases:
            cols = [c for c in df.columns if c.startswith(f"{base}_")]
            subset = cols #+ [c for c in ['num_voxels','species'] if c in df.columns]
            prefix = f"{base}_all_dims"
            print(f"\n=== Clustering: {prefix} ===")
            run_clustering(
                df[subset].copy(),
                drop_cols=['volume_physical','species','num_voxels', 'QSM_2d_error'],
                min_voxels=12,
                max_voxels=10000,
                z_thresh=3.5,
                n_pca=None,
                kmeans_k=2,
                kmeans_init='random',
                kmeans_n_init=20,
                kmeans_max_iter=500,
                meanshift_bw=None,
                meanshift_bin_seeding=True,
                meanshift_cluster_all=False,
                hdbscan_min_size=15,
                hdbscan_min_samples=20,
                hdbscan_epsilon=0.2,
                hdbscan_method='eom',
                umap_components=3,
                umap_n_neighbors=30,
                umap_min_dist=0.2,
                umap_metric='cosine',
                title_prefix=prefix,
                random_state=42
            )

        # 3) pairs of modalities by dimension
        for dim in dims:
            for m1, m2 in itertools.combinations(bases, 2):
                prefix = f"{m1}_{m2}_{dim}"
                cols = [c for c in df.columns if c.startswith(f"{m1}_{dim}_") or c.startswith(f"{m2}_{dim}_")]
                subset = cols #+ [c for c in ['num_voxels','species'] if c in df.columns]
                print(f"\n=== Clustering: {prefix} ===")
                run_clustering(
                    df[subset].copy(),
                    drop_cols=['volume_physical','species','num_voxels', 'QSM_2d_error'],
                    min_voxels=12,
                    max_voxels=10000,
                    z_thresh=3.5,
                    n_pca=None,
                    kmeans_k=2,
                    kmeans_init='random',
                    kmeans_n_init=20,
                    kmeans_max_iter=500,
                    meanshift_bw=None,
                    meanshift_bin_seeding=True,
                    meanshift_cluster_all=False,
                    hdbscan_min_size=5,
                    hdbscan_min_samples=10,
                    hdbscan_epsilon=0.2,
                    hdbscan_method='eom',
                    umap_components=3,
                    umap_n_neighbors=30,
                    umap_min_dist=0.2,
                    umap_metric='cosine',
                    title_prefix=prefix,
                    random_state=42
                )
    elif 'species' in df.columns:
        # Iris dataset pipeline
        print("\n=== Clustering Iris dataset ===")
        run_clustering(
            df,
            drop_cols=['species'],
            z_thresh=4.0,
            n_pca=2,
            kmeans_k=4,
            kmeans_init='k-means++',
            kmeans_n_init=10,
            kmeans_max_iter=300,
            meanshift_bw=None,
            meanshift_bin_seeding=True,
            meanshift_cluster_all=False,
            hdbscan_min_size=10,
            hdbscan_min_samples=None,
            hdbscan_epsilon=0.2,
            hdbscan_method='eom',
            umap_components=3,
            umap_n_neighbors=15,
            umap_min_dist=0.1,
            umap_metric='euclidean',
            title_prefix='Iris',
            random_state=42
        )
    else:
        print("No recognized target columns ('num_voxels' or 'species') found.")


if __name__ == "__main__":
    main()
