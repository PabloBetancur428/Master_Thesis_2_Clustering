import pandas as pd
import numpy as np

class FeatureFilter:
    """
    Filters out non-feature columns from the dataset.
    By default, removes 'label_id', 'PatientID', 'FolderType', and 'YearFolder'.
    """
    def __init__(self, drop_columns=None):
        if drop_columns is None:
            drop_columns = ['label_id', 'PatientID', 'FolderType', 'YearFolder']
        self.drop_columns = drop_columns

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        features = data.drop(columns=[col for col in self.drop_columns if col in data.columns])
        # Keep only numeric columns for clustering.
        return features.select_dtypes(include=[np.number])
