
# feature_filter.py
from typing import List
import pandas as pd

class FeatureFilter:
    """
    Filters DataFrame columns to retain only numeric features or a selected subset.

    Attributes:
        drop_columns (List[str]): Columns to remove before numeric filtering.
        select_columns (List[str]): Explicit columns to select (overrides drop_columns).
    """
    def __init__(self, drop_columns: List[str] = None, select_columns: List[str] = None):
        """
        Configures which columns to drop or select.

        Args:
            drop_columns (List[str], optional): Column names to remove.
            select_columns (List[str], optional): Column names to keep.
        """
        self.drop_columns = drop_columns or []
        self.select_columns = select_columns or []

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies column filtering and returns only numeric data.

        Args:
            df (pd.DataFrame): Input DataFrame to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame containing numeric features.
        """
        if self.select_columns:
            return df[self.select_columns].copy()
        return df.drop(columns=self.drop_columns, errors='ignore')._get_numeric_data()