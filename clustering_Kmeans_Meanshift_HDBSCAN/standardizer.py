import pandas as pd
from sklearn.preprocessing import StandardScaler

class Standardizer:
    """
    Fit a StandardScaler to a DataFrame and transform it, preserving indices and column names.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> 'Standardizer':
        """
        Fit the scaler using the DataFrame's numeric values.
        """
        self.scaler.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame to zero mean and unit variance, returning a DataFrame
        with the original index and column names.
        """
        scaled_array = self.scaler.transform(df)
        return pd.DataFrame(scaled_array, index=df.index, columns=df.columns)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler and transform the DataFrame in one step.
        """
        return self.fit(df).transform(df)
