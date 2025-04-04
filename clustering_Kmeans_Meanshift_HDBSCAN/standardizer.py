from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class Standardizer:
    """
    Standardizes the data using z-score normalization.
    """
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(data.values)
