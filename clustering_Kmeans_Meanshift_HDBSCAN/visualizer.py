import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting support

class Visualizer:
    """
    Provides static methods to visualize clustering results.
    This version automatically chooses between 2D and 3D plotting based on the data.
    """
    @staticmethod
    def plot_clusters(reduced_data: np.ndarray, labels: np.ndarray, title: str = "Cluster Plot") -> None:
        if reduced_data.shape[1] == 3:
            # Create a 3D scatter plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                                 c=labels, cmap='viridis', s=50, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.set_zlabel("Dimension 3")
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.1)
            cbar.set_label("Cluster Label")
            plt.grid(True)
            plt.savefig(f"{title}.png")
            plt.show()
        else:
            # Create a 2D scatter plot
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                                  c=labels, cmap='viridis', s=50, alpha=0.7)
            plt.title(title)
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.colorbar(scatter, label="Cluster Label")
            plt.grid(True)
            plt.savefig(f"{title}.png")
            plt.show()
