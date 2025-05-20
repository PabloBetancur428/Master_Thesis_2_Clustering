import pandas as pd
from umap.umap_ import UMAP

class UMAPVisualizer:
    """
    Wraps UMAP from umap-learn, preserving DataFrame index and returning a DataFrame.

    Attributes:
        reducer (UMAP): underlying UMAP model.
    """
    def __init__(self, n_components: int = 2, random_state: int = None, **kwargs):
        """
        Initializes UMAP reducer with given parameters.

        Args:
            n_components (int): target dimensionality.
            random_state (int): random seed.
            **kwargs: additional UMAP parameters (e.g., n_neighbors, min_dist, metric).
        """
        self.reducer = UMAP(n_components=n_components, random_state=random_state, **kwargs)

    def fit(self, X: pd.DataFrame) -> 'UMAPVisualizer':
        """
        Fit the UMAP model to the DataFrame X.

        Args:
            X (pd.DataFrame): input features.
        Returns:
            self
        """
        self.reducer.fit(X.values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X into UMAP embed, preserving index and returning a DataFrame.

        Args:
            X (pd.DataFrame): input features.
        Returns:
            pd.DataFrame: embedding with columns ['UMAP1', 'UMAP2', ...].
        """
        embedding = self.reducer.transform(X.values)
        cols = [f"UMAP{i+1}" for i in range(embedding.shape[1])]
        return pd.DataFrame(embedding, index=X.index, columns=cols)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit UMAP and transform X in one step.

        Args:
            X (pd.DataFrame): input features.
        Returns:
            pd.DataFrame: embedding.
        """
        embedding = self.reducer.fit_transform(X.values)
        cols = [f"UMAP{i+1}" for i in range(embedding.shape[1])]
        return pd.DataFrame(embedding, index=X.index, columns=cols)
