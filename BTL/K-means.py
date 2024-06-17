import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import DataProcessor
import matplotlib.pyplot as plt

X_train, y_train, X_val, y_val = DataProcessor.train_test_data()

X_combined = np.concatenate((X_train, X_val), axis=0)

X_combined_reshaped = X_combined.reshape(X_combined.shape[0], -1)

kmeans_combined = KMeans(n_clusters=3, random_state=42)
kmeans_combined.fit(X_combined_reshaped)

pca_2d = PCA(n_components=2)
X_combined_pca_2d = pca_2d.fit_transform(X_combined_reshaped)

pca_3d = PCA(n_components=3)
X_combined_pca_3d = pca_3d.fit_transform(X_combined_reshaped)

cluster_labels = ['cluster 1', 'cluster 2', 'cluster 3']


def plot_pca_2d(data, labels, title, cluster_labels):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.legend(handles=scatter.legend_elements()[0], labels=cluster_labels, title="Clusters")
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


def plot_pca_3d(data, labels, title, cluster_labels):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis')
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.show()

plot_pca_2d(X_combined_pca_2d, kmeans_combined.labels_, 'Combined Data Cluster 2D', cluster_labels)

plot_pca_3d(X_combined_pca_3d, kmeans_combined.labels_, 'Combined Data Cluster 3D', cluster_labels)
