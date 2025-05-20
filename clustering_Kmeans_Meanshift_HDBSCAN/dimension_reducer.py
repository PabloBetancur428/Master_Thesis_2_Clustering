import pandas as pd
from sklearn.decomposition import PCA

class ClusterReducer:
    """
    Reduce dimensionality of a DataFrame using PCA, preserving indices.

    Attributes:
        n_components (int): Number of principal components.
        random_state (int): Seed for reproducibility.
    """
    def __init__(self, n_components: int = 2, random_state: int = None):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = PCA(n_components=n_components, random_state=random_state)

    def fit(self, df: pd.DataFrame) -> 'ClusterReducer':
        """
        Fit the PCA model to the DataFrame.
        """
        self.reducer.fit(df.values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame into principal components, returning a DataFrame
        with preserved index and new component column names.
        """
        reduced = self.reducer.transform(df.values)
        # Build column names: PC1, PC2, ...
        cols = [f"PC{i+1}" for i in range(reduced.shape[1])]
        return pd.DataFrame(reduced, index=df.index, columns=cols)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit PCA to the DataFrame and transform it in one step.
        """
        self.fit(df)
        return self.transform(df)
