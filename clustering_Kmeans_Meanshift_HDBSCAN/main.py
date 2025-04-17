from data_loader import DataLoader
from outlier_remover import OutlierRemover
from feature_filter import FeatureFilter
from standardizer import Standardizer
from dimension_reducer import ClusterReducer
from clusterers import KMeansClusterer, MeanShiftClusterer, HDBSCANClusterer
from visualizer import Visualizer
from umap_visualizer import UMAPVisualizer
import numpy as np
import pandas as pd

def main():
    # Define the path to your iris dataset.
    dataset_path = "/home/jbetancur/Desktop/codes/clustering/data/archive/iris.csv" 
    
    # Data Loading
    loader = DataLoader(dataset_path)
    data = loader.load()
    
    # Assume the iris dataset has columns:
    # ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    # Save ground truth species labels for later comparison.
    ground_truth = data['species']
    numeric_labels, uniques = pd.factorize(ground_truth)
    # Make sure numeric_labels is an integer numpy array
    numeric_labels = np.array(numeric_labels, dtype=int)
    
    # Feature Filtering: Drop the target column 'species' so that only numeric features are used.
    filterer = FeatureFilter(drop_columns=['species'])
    features = filterer.filter(data)
    print(f"Initial dataset features shape: {features.shape}")
    
    # Outlier Removal: Remove extreme values if any.
    outlier_remover = OutlierRemover(z_thresh=4.0)
    features_no_outliers = outlier_remover.remove_outliers(features)
    print("After outlier removal:", features_no_outliers.shape)
    
    # Standardization: Scale the features.
    standardizer = Standardizer()
    scaled_data = standardizer.fit_transform(features_no_outliers)
    
    # Dimensionality Reduction for Clustering.
    # For iris, 4 principal components suffice; adjust n_components if desired.
    cluster_reducer = ClusterReducer(n_components=4)
    cluster_data = cluster_reducer.reduce(scaled_data)
    
    # Clustering with three different algorithms.
    # K-Means (we set k=3 because there are three iris species).
    kmeans_clusterer = KMeansClusterer(n_clusters=3)
    kmeans_labels = kmeans_clusterer.fit_predict(cluster_data)
    
    # MeanShift Clustering
    meanshift_clusterer = MeanShiftClusterer()
    meanshift_labels = meanshift_clusterer.fit_predict(cluster_data)
    
    # HDBSCAN Clustering
    hdbscan_clusterer = HDBSCANClusterer(min_cluster_size=5)
    hdbscan_labels = hdbscan_clusterer.fit_predict(cluster_data)
    
    print(f"KMeans found {len(np.unique(kmeans_labels))} clusters")
    print(f"MeanShift found {len(np.unique(meanshift_labels))} clusters")
    print(f"HDBSCAN found {len(np.unique(hdbscan_labels))} clusters")
    
    # UMAP Visualization: Project the reduced (cluster_data) into 3 dimensions.
    umap_viz = UMAPVisualizer(n_components=3, random_state=42)
    umap_data = umap_viz.reduce(cluster_data)
    
    # Plot the ground truth species for reference using the new 3D plot.
    Visualizer.plot_clusters(umap_data, numeric_labels, title="Ground Truth - Iris Species")
    
    # Plot the clustering results from each algorithm.
    Visualizer.plot_clusters(umap_data, kmeans_labels, title="K-Means Clustering")
    Visualizer.plot_clusters(umap_data, meanshift_labels, title="MeanShift Clustering")
    Visualizer.plot_clusters(umap_data, hdbscan_labels, title="HDBSCAN Clustering")

if __name__ == "__main__":
    main()
