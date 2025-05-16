import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    @staticmethod
    def plot_clusters(X, labels, title: str = None):
        """
        X: numpy array of shape (n_samples, 2) or (n_samples, 3)
        labels: array-like cluster labels
        """
        dim = X.shape[1]
        fig = plt.figure()
        if dim == 2:
            ax = fig.add_subplot(1,1,1)
            scatter = ax.scatter(X[:,0], X[:,1], c=labels)
        elif dim == 3:
            ax = fig.add_subplot(1,1,1, projection='3d')
            scatter = ax.scatter(X[:,0], X[:,1], X[:,2], c=labels)
        if title:
            plt.title(title)
        plt.show()