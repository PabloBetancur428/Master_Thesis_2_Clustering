# standardizer.py
from sklearn.preprocessing import StandardScaler

class Standardizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, df):
        """
        Fit to data, then transform it to zero mean, unit variance.
        """
        return self.scaler.fit_transform(df)