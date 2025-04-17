import umap
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection

class UMAPVisualizer:
    """
    Provides methods to reduce dimensionality using UMAP and plot the resulting clusters.
    """
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        self.n_components = n_components  # Save the number of components for later check.
    
    def reduce(self, data: np.ndarray) -> np.ndarray:
        """
        Reduces high-dimensional data to a lower-dimensional space using UMAP.
        
        Parameters:
            data (np.ndarray): The high-dimensional data.
        
        Returns:
            np.ndarray: The data transformed into the lower-dimensional space.
        """
        return self.reducer.fit_transform(data)
    
    def plot_clusters(self, reduced_data: np.ndarray, labels: np.ndarray, title: str = "UMAP Cluster Plot") -> None:
        """
        Plots the clusters using the UMAP-reduced data.
        
        Parameters:
            reduced_data (np.ndarray): The data after UMAP reduction.
            labels (np.ndarray): Cluster labels (or numeric values) for each data point.
            title (str, optional): The title of the plot.
        """
        if self.n_components == 3 and reduced_data.shape[1] == 3:
            # Create a 3D scatter plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                                 c=labels, cmap='viridis', s=50, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            fig.colorbar(scatter, ax=ax, label="Cluster Label")
        else:
            # Default 2D scatter plot
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                                  c=labels, cmap='viridis', s=50, alpha=0.7)
            plt.title(title)
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plt.colorbar(scatter, label="Cluster Label")
            plt.grid(True)
        
        print("Saving UMAP plot...")
        plt.savefig(f"UMAP_{title}.png")
        plt.show()
