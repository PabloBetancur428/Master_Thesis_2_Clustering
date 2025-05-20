import numpy as np
import pandas as pd

class OutlierRemover:
    """
    Removes rows where any numeric feature's z-score exceeds a threshold.

    Attributes:
        z_thresh (float): Threshold for absolute z-score to consider an outlier.
    """
    def __init__(self, z_thresh: float = 3.0):
        """
        Initializes the remover with a z-score threshold.

        Args:
            z_thresh (float): |z-score| cutoff; typical range 2.5–4.0.
        """
        self.z_thresh = z_thresh

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes z-scores and filters out rows exceeding the threshold.

        Args:
            df (pd.DataFrame): DataFrame with numeric columns.

        Returns:
            pd.DataFrame: DataFrame without outlier rows, preserving original index.
        """
        # Select numeric columns for z-score calculation
        numeric = df.select_dtypes(include=[np.number])
        # Calculate z-scores per column
        z_scores = (numeric - numeric.mean()) / numeric.std(ddof=0)
        # Mask rows where all numeric values are within threshold
        mask = (z_scores.abs() <= self.z_thresh).all(axis=1)
        # Return filtered DataFrame, preserving original index for alignment
        return df.loc[mask]
