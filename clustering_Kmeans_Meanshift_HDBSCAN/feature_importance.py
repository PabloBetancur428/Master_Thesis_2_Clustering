import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
def compute_feature_importance(X: pd.DataFrame, labels: np.ndarray, method: str = 'forest') -> pd.DataFrame:
    """
    Computes feature importance scores based on clustering labels.

    Args:
        X (pd.DataFrame): The feature matrix used in clustering.
        labels (np.ndarray): Cluster labels to treat as target variable.
        method (str): 'forest' for RandomForest importance, 'permutation' for permutation importance.

    Returns:
        pd.DataFrame: DataFrame with features and importance scores sorted descending.
    """
    if method == 'forest':
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(X, labels)
        importances = model.feature_importances_
    else:
        # use a baseline forest to measure permutation importance
        base = RandomForestClassifier(n_estimators=100, random_state=0)
        base.fit(X, labels)
        result = permutation_importance(base, X, labels, n_repeats=10, random_state=0)
        importances = result.importances_mean

    feat_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    return feat_imp


def plot_feature_importance(feat_imp: pd.DataFrame, title: str = None):
    """
    Plots feature importance as a horizontal bar chart.

    Args:
        feat_imp (pd.DataFrame): DataFrame with 'feature' and 'importance'.
        title (str, optional): Title for the plot.
    """
    plt.figure(figsize=(8, len(feat_imp) * 0.3 + 1))
    plt.barh(feat_imp['feature'], feat_imp['importance'])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()