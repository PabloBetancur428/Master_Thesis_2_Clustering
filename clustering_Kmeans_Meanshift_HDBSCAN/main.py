from data_loader import DataLoader
from outlier_remover import OutlierRemover
from feature_filter import FeatureFilter
from standardizer import Standardizer
from dimension_reducer import ClusterReducer, VisualizationReducer
from clusterers import KMeansClusterer, MeanShiftClusterer, HDBSCANClusterer
from visualizer import Visualizer
from umap_visualizer import UMAPVisualizer
import numpy as np

def main():
    # Define the path to your dataset
    dataset_path = "clustering_Kmeans_Meanshift_HDBSCAN/Cleaned_dataset_4now.csv" 
    # Data Loading
    loader = DataLoader(dataset_path)
    data = loader.load()
    
    # Feature Filtering
    filterer = FeatureFilter()
    features = filterer.filter(data)
    
    #Outlier removal
    outlier_remover = OutlierRemover(z_thresh=3.0)
    features_no_outliers = outlier_remover.remove_outliers(features)

    # Standardization
    standardizer = Standardizer()
    scaled_data = standardizer.fit_transform(features_no_outliers)
    
    # Dimensionality Reduction for Clustering (retain more variance, e.g., 10 components)
    cluster_reducer = ClusterReducer(n_components=20)
    cluster_data = cluster_reducer.reduce(scaled_data)
    
    # Clustering on the higher-dimensional space
    kmeans_clusterer = KMeansClusterer(n_clusters=2)
    kmeans_labels = kmeans_clusterer.fit_predict(cluster_data)
    
    meanshift_clusterer = MeanShiftClusterer()
    meanshift_labels = meanshift_clusterer.fit_predict(cluster_data)
    
    hdbscan_clusterer = HDBSCANClusterer(min_cluster_size=5)
    hdbscan_labels = hdbscan_clusterer.fit_predict(cluster_data)
    
    # UMAP Visualization
    umap_viz = UMAPVisualizer(n_components=2, random_state=42)
    umap_data = umap_viz.reduce(cluster_data)

    
    # Plot the results using the 2D projection
    Visualizer.plot_clusters(umap_data, kmeans_labels, title="K-Means Clustering")
    Visualizer.plot_clusters(umap_data, meanshift_labels, title="MeanShift Clustering")
    Visualizer.plot_clusters(umap_data, hdbscan_labels, title="HDBSCAN Clustering")


    
if __name__ == "__main__":
    main()
