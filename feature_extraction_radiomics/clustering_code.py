"""
clustering.py

Module for clustering radiomic feature vectors using K-means.
"""

import numpy as np
from sklearn.cluster import KMeans

class KMeansCluster:
    def __init__(self, n_clusters=2, random_state=42):
        """
        Initialize the K-means clusterer.

        Parameters:
            n_clusters (int): Number of clusters.
            random_state (int): Seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

    def fit(self, feature_matrix):
        """
        Fit the K-means model to the given feature matrix.

        Parameters:
            feature_matrix (np.ndarray): 2D array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Cluster labels for each sample.
        """
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = self.model.fit_predict(feature_matrix)
        return labels
