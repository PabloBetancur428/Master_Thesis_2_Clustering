import pandas as pd
import numpy as np
from scipy import stats

class OutlierRemover:
    """
    Provides methods to remove outliers from a DataFrame using the Z-score method.
    """
    def __init__(self, z_thresh: float = 3.0):
        """
        Parameters:
            z_thresh: The threshold for the Z-score beyond which data points are considered outliers.
        """
        self.z_thresh = z_thresh

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows from the DataFrame where any numeric column has a Z-score above the threshold.
        
        Parameters:
            df: The input DataFrame.
            
        Returns:
            A new DataFrame with outlier rows removed.
        """
        # Work only on numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        # Compute the Z-scores
        z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
        # Identify rows where all Z-scores are below the threshold
        filtered_entries = (z_scores < self.z_thresh).all(axis=1)
        # Return a new DataFrame with these rows only
        return df[filtered_entries].reset_index(drop=True)
