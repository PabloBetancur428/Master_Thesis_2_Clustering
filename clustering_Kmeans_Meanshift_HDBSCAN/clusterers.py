import numpy as np
from sklearn.cluster import KMeans, MeanShift
import hdbscan

class KMeansClusterer:
    """
    Applies K-Means clustering.
    """
    def __init__(self, n_clusters: int = 2, random_state: int = 42):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
    
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(data)

class MeanShiftClusterer:
    """
    Applies MeanShift clustering.
    """
    def __init__(self):
        self.model = MeanShift()
    
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(data)

class HDBSCANClusterer:
    """
    Applies HDBSCAN clustering.
    """
    def __init__(self, min_cluster_size: int = 5):
        self.model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(data)
