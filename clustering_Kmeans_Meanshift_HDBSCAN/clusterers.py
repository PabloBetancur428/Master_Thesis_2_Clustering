# clusterers.py
from sklearn.cluster import KMeans, MeanShift
import hdbscan

class KMeansClusterer:
    """
    Wrapper for KMeans clustering.

    Attributes:
        model (KMeans): Internal model.
    """
    def __init__(self,
                 n_clusters: int = 3,
                 init: str = 'k-means++',
                 n_init: int = 10,
                 max_iter: int = 300,
                 random_state: int = None):
        """
        Args:
            n_clusters (int): Number of clusters to form. >0; typical 2–10.
            init (str): Method for initialization 'k-means++' or 'random'. Defaults to 'k-means++'.
            n_init (int): Number of time the k-means algorithm will run with different centroid seeds. >0; typical 10–50. Defaults to 10.
            max_iter (int): Maximum iterations per run. >0; typical 300–1000. Defaults to 300.
            random_state (int, optional): Seed; None or int. Defaults to None.
        """
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )

    def fit_predict(self, X):
        """
        Args:
            X (np.ndarray or pd.DataFrame): Data.

        Returns:
            np.ndarray: Cluster labels.
        """
        return self.model.fit_predict(X)

class MeanShiftClusterer:
    """
    Wrapper for MeanShift clustering.

    Attributes:
        model (MeanShift): Internal model.
    """
    def __init__(
        self,
        bandwidth: float = None,
        bin_seeding: bool = False,
        cluster_all: bool = True
    ):
        """
        Args:
            bandwidth (float, optional): Kernel bandwidth. >0; if None, estimated via sklearn.estimate_bandwidth(X, quantile).
            bin_seeding (bool): True speeds up by seeding bins. Defaults to False.
            cluster_all (bool): True assigns all points; False leaves some as noise. Defaults to True.
        """
        self.model = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=bin_seeding,
            cluster_all=cluster_all
        )

    def fit_predict(self, X):
        """
        Args:
            X (np.ndarray or pd.DataFrame): Data.

        Returns:
            np.ndarray: Cluster labels.
        """
        return self.model.fit_predict(X)

class HDBSCANClusterer:
    """
    Wrapper for HDBSCAN clustering.

    Attributes:
        model (hdbscan.HDBSCAN): Internal model.
    """
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = 'eom',
        **kwargs
    ):
        """
        Args:
            min_cluster_size (int): Minimum size for clusters. >1; typical 3–50. Defaults to 5.
            min_samples (int, optional): Minimum samples in a neighborhood for a point to be core. >0; defaults to min_cluster_size if None.
            cluster_selection_epsilon (float): Distance threshold to split clusters. >=0; small splits clusters. Defaults to 0.0.
            cluster_selection_method (str): excess of mass ('eom')  or leaf selection ('leaf'). Defaults to 'eom'.
            **kwargs: Other HDBSCAN settings.
        """
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            **kwargs
        )

    def fit_predict(self, X):
        """
        Args:
            X (np.ndarray or pd.DataFrame): Data.

        Returns:
            np.ndarray: Cluster labels.
        """
        return self.model.fit_predict(X)