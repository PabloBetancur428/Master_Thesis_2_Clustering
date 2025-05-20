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
) -> pd.DataFrame:
    """
    Executes the clustering pipeline and returns a DataFrame with cluster labels.
    DataFrame index aligns with the input `data` index, so metadata can be joined easily.
    """
    # Track original index
    idx = data.index

    # Step 1: optional filter by voxel count
    print(f"Before voxel filtering ({title_prefix}): {data.shape}")
    if 'num_voxels' in data.columns:
        data = filter_by_voxel_count(data, min_voxels, max_voxels)
    print(f"After voxel filtering ({title_prefix}): {data.shape}")

    # Step 2: extract ground truth (if available)
    truth = None
    if 'species' in data.columns:
        truth = pd.factorize(data['species'])[0]

    # Step 3: feature filtering
    features = FeatureFilter(drop_columns=drop_cols).filter(data)
    print(f"Features shape ({title_prefix}): {features.shape}")

    # Step 4: outlier removal (currently disabled)
    clean = features.copy()

    # Step 5: standardization
    scaled = Standardizer().fit_transform(clean)

    # Step 6: PCA reduction
    pca_df = ClusterReducer(n_components=n_pca, random_state=random_state).fit_transform(scaled)

    # Step 7: clustering models
    km_series = KMeansClusterer(
        n_clusters=kmeans_k,
        init=kmeans_init,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
        random_state=random_state
    ).fit_predict(pca_df)
    ms_series = MeanShiftClusterer(
        bandwidth=meanshift_bw,
        bin_seeding=meanshift_bin_seeding,
        cluster_all=meanshift_cluster_all
    ).fit_predict(pca_df)
    hb_series = HDBSCANClusterer(
        min_cluster_size=hdbscan_min_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_epsilon=hdbscan_epsilon,
        cluster_selection_method=hdbscan_method
    ).fit_predict(pca_df)

    # Step 8: print cluster counts
    print("------------------------------------------")
    print(f"KMeans clusters ({title_prefix}): {len(np.unique(km_series))}")
    print(f"MeanShift clusters ({title_prefix}): {len(np.unique(ms_series))}")
    print(f"HDBSCAN clusters ({title_prefix}): {len(np.unique(hb_series))}")

    # Step 9: UMAP visualization
    umap_df = UMAPVisualizer(
        n_components=umap_components,
        random_state=random_state,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric
    ).fit_transform(pca_df)

    # Step 10: plotting
    prefix = f"{title_prefix} - " if title_prefix else ''
    if truth is not None:
        Visualizer.plot_clusters(umap_df, truth, title=f"{prefix}Ground Truth")
    Visualizer.plot_clusters(umap_df, km_series, title=f"{prefix}KMeans")
    Visualizer.plot_clusters(umap_df, ms_series, title=f"{prefix}MeanShift")
    Visualizer.plot_clusters(umap_df, hb_series, title=f"{prefix}HDBSCAN")

    # Compile labels into a DataFrame
    labels_df = pd.concat([km_series, ms_series, hb_series], axis=1)
    return labels_df


def main():
    data_path = "/home/jbetancur/Desktop/codes/clustering/feature_extraction/output/Radiomic_QSM_Wavelet_LBP_2d_3d.csv"
    # Load only first 10 rows for testing, then slice full dataset when ready
    df = DataLoader(data_path).load().head(10)

    # Separate metadata
    meta = df[['label_id', 'PatientID']].copy()
    df = df.drop(columns=['label_id', 'PatientID'])

    all_results = []

    if 'num_voxels' in df.columns:
        full_mods = sorted({
            "_".join(c.split('_')[:2])
            for c in df.columns
            if '_' in c and c not in ['num_voxels','species','label_id','volume_physical']
        })

        for mod in full_mods:
            if "2d" in mod:
                continue
            cols = [c for c in df.columns if c.startswith(f"{mod}_")]
            print(f"\n=== Clustering modality: {mod} ===")
            # Retain raw features for later analysis
            features_df = df[cols].copy()
            # Run clustering to get labels
            labels_df = run_clustering(
                features_df,
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
            # Combine metadata, raw features, and cluster labels
            result = pd.concat([meta, features_df, labels_df], axis=1)
            result['modality'] = mod
            all_results.append(result)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        print("Clustering complete. Final results shape:", final_df.shape)
        # Save enriched table with features and labels
        final_df.to_csv("clustered_lesions_with_features_and_meta.csv", index=False)
    else:
        print("No clustering performed.")


if __name__ == "__main__":
    main()
