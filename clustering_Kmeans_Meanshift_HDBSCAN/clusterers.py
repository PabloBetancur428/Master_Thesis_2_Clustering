import pandas as pd
from sklearn.cluster import KMeans, MeanShift
import hdbscan

class KMeansClusterer:
    def __init__(self, n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=None):
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Fit KMeans and return cluster labels as a pandas Series aligned to X's index.
        """
        labels = self.model.fit_predict(X.values)
        return pd.Series(labels, index=X.index, name='kmeans_label')

class MeanShiftClusterer:
    def __init__(self, bandwidth=None, bin_seeding=False, cluster_all=True):
        self.model = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=bin_seeding,
            cluster_all=cluster_all
        )

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Fit MeanShift and return cluster labels as a pandas Series aligned to X's index.
        """
        labels = self.model.fit_predict(X.values)
        return pd.Series(labels, index=X.index, name='meanshift_label')

class HDBSCANClusterer:
    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, cluster_selection_method='eom'):
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method
        )

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Fit HDBSCAN and return cluster labels as a pandas Series aligned to X's index.
        """
        labels = self.model.fit_predict(X.values)
        return pd.Series(labels, index=X.index, name='hdbscan_label')
