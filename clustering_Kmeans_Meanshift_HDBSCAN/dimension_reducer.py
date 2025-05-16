# dimension_reducer.py
from sklearn.decomposition import PCA

class ClusterReducer:
    def __init__(self, n_components: int = 2, random_state: int = None):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = PCA(n_components=n_components, random_state=random_state)

    def reduce(self, X):
        """
        Reduce dimensionality of numeric array X.
        """
        return self.reducer.fit_transform(X)