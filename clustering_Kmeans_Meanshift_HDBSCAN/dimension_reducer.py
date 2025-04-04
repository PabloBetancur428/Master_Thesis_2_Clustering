from sklearn.decomposition import PCA
import numpy as np

class ClusterReducer:
    """
    Reduces the data dimensionality for clustering purposes.
    A higher number of components (default is 10) is used to retain more variance.
    """
    def __init__(self, n_components: int = 10, random_state: int = 42):
        self.pca = PCA(n_components=n_components, random_state=random_state)
    
    def reduce(self, data: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(data)

class VisualizationReducer:
    """
    Reduces the data dimensionality for visualization purposes.
    Typically reduced to 2 dimensions.
    """
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.pca = PCA(n_components=n_components, random_state=random_state)
    
    def reduce(self, data: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(data)
