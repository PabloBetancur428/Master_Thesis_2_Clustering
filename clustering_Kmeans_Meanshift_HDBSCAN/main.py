from data_loader import DataLoader
from outlier_remover import OutlierRemover
from feature_filter import FeatureFilter
from standardizer import Standardizer
from dimension_reducer import ClusterReducer
from clusterers import KMeansClusterer, MeanShiftClusterer, HDBSCANClusterer
from visualizer import Visualizer
from umap_visualizer import UMAPVisualizer
import numpy as np

def main():
    # Define the path to your dataset
    dataset_path = "clustering_Kmeans_Meanshift_HDBSCAN/patients_clean_radiomics_features.csv" 
    # Data Loading
    loader = DataLoader(dataset_path)
    data = loader.load()
    
    # Feature Filtering: drop non-feature columns
    filterer = FeatureFilter(drop_columns=['label_id', 'PatientID', 'FolderType', 'YearFolder'])
    features = filterer.filter(data)
    print(f"Initial dataset is of shape: {features.shape}")
    
    # Define the modalities to process based on column name prefixes.
    modalities = ["T1", "T2", "QSM"]
    
    # Create common instances for outlier removal and dimensionality reduction
    outlier_remover = OutlierRemover(z_thresh=3.0)
    standardizer = Standardizer()
    cluster_reducer = ClusterReducer(n_components=200)
    umap_viz = UMAPVisualizer(n_components=3, random_state=42)
    
    # Loop over each modality
    for mod in modalities:
        print(f"\nProcessing modality: {mod}")
        
        # Select only columns for the current modality (e.g., keys starting with "T1_")
        mod_features = features.filter(regex=f"^{mod}_")
        print(f"{mod} features shape: {mod_features.shape}")
        
        # Outlier removal
        mod_features_no_outliers = outlier_remover.remove_outliers(mod_features)
        
        # Standardization
        mod_scaled_data = standardizer.fit_transform(mod_features_no_outliers)
        
        # Dimensionality Reduction for Clustering
        mod_cluster_data = cluster_reducer.reduce(mod_scaled_data)
        
        # Perform clustering with three clusterers
        # 1. K-Means Clustering
        kmeans_clusterer = KMeansClusterer(n_clusters=2)
        kmeans_labels = kmeans_clusterer.fit_predict(mod_cluster_data)
        
        # 2. MeanShift Clustering
        meanshift_clusterer = MeanShiftClusterer()
        meanshift_labels = meanshift_clusterer.fit_predict(mod_cluster_data)
        
        # 3. HDBSCAN Clustering
        hdbscan_clusterer = HDBSCANClusterer(min_cluster_size=10)
        hdbscan_labels = hdbscan_clusterer.fit_predict(mod_cluster_data)
        
        # Report number of clusters for each algorithm
        print(f"{mod} KMeans: {len(np.unique(kmeans_labels))} clusters")
        print(f"{mod} MeanShift: {len(np.unique(meanshift_labels))} clusters")
        print(f"{mod} HDBSCAN: {len(np.unique(hdbscan_labels))} clusters")
        
        # UMAP Visualization: reduce to 3D for plotting
        mod_umap_data = umap_viz.reduce(mod_cluster_data)
        
        # Plot each clustering result
        Visualizer.plot_clusters(mod_umap_data, kmeans_labels, title=f"{mod} K-Means Clustering")
        Visualizer.plot_clusters(mod_umap_data, meanshift_labels, title=f"{mod} MeanShift Clustering")
        Visualizer.plot_clusters(mod_umap_data, hdbscan_labels, title=f"{mod} HDBSCAN Clustering")

if __name__ == "__main__":
    main()
