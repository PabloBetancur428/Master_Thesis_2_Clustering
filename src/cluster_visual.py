import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class ClusterVisualizer:
    """
    A class for visualizing clustering results.
    
    This class reduces the feature matrix to 2 dimensions using PCA and 
    produces a scatter plot where each point is annotated with its lesion ID.
    """
    
    def __init__(self, cmap='tab10'):
        """
        Initialize the visualizer.
        
        Parameters:
            cmap (str): Colormap to use for the scatter plot.
        """
        self.cmap = cmap

    def plot_clusters(self, feature_matrix, labels, lesion_ids, title="Lesion Clusters (PCA)"):
        """
        Reduce the feature matrix to 2D using PCA and plot the clusters.

        Parameters
        ----------
        feature_matrix : np.ndarray
            2D array of shape (n_samples, n_features) containing the features.
        labels : np.ndarray or list
            Cluster labels for each sample.
        lesion_ids : list
            List of lesion IDs corresponding to each sample.
        title : str
            Title of the plot.
        """
        # Reduce dimensionality to 2 components for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(feature_matrix)

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                              c=labels, cmap=self.cmap, s=100, alpha=0.8)

        # Annotate each point with the lesion id
        for i, lesion_id in enumerate(lesion_ids):
            plt.annotate(str(lesion_id), (features_2d[i, 0], features_2d[i, 1]),
                         textcoords="offset points", xytext=(5,5), fontsize=9)

        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster Label")
        plt.tight_layout()
        plt.savefig("first_cluster.png")
