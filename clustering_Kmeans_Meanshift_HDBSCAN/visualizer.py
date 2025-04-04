import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """
    Provides static methods to visualize clustering results.
    """
    @staticmethod
    def plot_clusters(reduced_data: np.ndarray, labels: np.ndarray, title: str = "Cluster Plot") -> None:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster Label")
        plt.grid(True)
        plt.savefig(f"{title}.png")
