from umap.umap_ import UMAP

class UMAPVisualizer:
    def __init__(self, n_components: int = 2, random_state: int = None, **kwargs):
        """
        Wraps UMAP from umap-learn. Pass additional UMAP parameters via kwargs.
        """
        self.reducer = UMAP(n_components=n_components, random_state=random_state, **kwargs)

    def reduce(self, X):
        return self.reducer.fit_transform(X)