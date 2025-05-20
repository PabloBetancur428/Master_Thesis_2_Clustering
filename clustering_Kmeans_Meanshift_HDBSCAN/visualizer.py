import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

class Visualizer:
    @staticmethod
    def plot_clusters(X, labels, title: str = None):
        """
        Plot clusters in 2D or 3D. Accepts X as DataFrame or array, and labels as array-like or Series.

        Args:
            X (pd.DataFrame or np.ndarray): Input data with shape (n_samples, 2) or (n_samples, 3).
            labels (pd.Series or array-like): Cluster labels for coloring.
            title (str, optional): Plot title.
        """
        # Extract numeric array from DataFrame if needed
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = np.asarray(X)

        # Extract label values
        if isinstance(labels, pd.Series):
            label_vals = labels.values
        else:
            label_vals = np.asarray(labels)

        dim = data.shape[1]
        fig = plt.figure()
        if dim == 2:
            ax = fig.add_subplot(1, 1, 1)
            scatter = ax.scatter(data[:, 0], data[:, 1], c=label_vals)
        elif dim == 3:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=label_vals)
        else:
            raise ValueError(f"Visualizer supports only 2D or 3D data, got {dim}D.")

        if title:
            ax.set_title(title)
        plt.show()
