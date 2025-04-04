import os
import pandas as pd

class DataLoader:
    """
    Loads a CSV dataset from a given file path.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        return pd.read_csv(self.dataset_path)