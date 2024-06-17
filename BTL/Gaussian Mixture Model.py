import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import DataProcessor


def plot_gmm(X_train, y_train, X_val, y_val, title, n_components_pca=2, n_components_gmm=3):
    # Combine the train and validation data
    X_combined = np.concatenate((X_train, X_val), axis=0)
    y_combined = np.concatenate((y_train, y_val), axis=0)

    # Reshape the data
    X_reshaped = X_combined.reshape(X_combined.shape[0], -1)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # Perform PCA
    pca = PCA(n_components=n_components_pca)
    X_pca = pca.fit_transform(X_scaled)

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components_gmm)
    gmm.fit(X_pca)
    y_gmm = gmm.predict(X_pca)

    # Plotting
    if n_components_pca == 2:
        plt.figure(figsize=(8, 6))
        for class_label in np.unique(y_gmm):
            plt.scatter(X_pca[y_gmm == class_label, 0], X_pca[y_gmm == class_label, 1], label=f'Cluster {class_label}')
        plt.title(title + ' - 2D')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.show()

    elif n_components_pca == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for class_label in np.unique(y_gmm):
            ax.scatter(X_pca[y_gmm == class_label, 0], X_pca[y_gmm == class_label, 1], X_pca[y_gmm == class_label, 2],
                       label=f'Cluster {class_label}')
        ax.set_title(title + ' - 3D')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend(title='Cluster')
        plt.show()


if __name__ == '__main__':
    X_train, y_train, X_val, y_val = DataProcessor.train_test_data()

    # Plotting GMM for the combined training and validation data in 2D and 3D
    plot_gmm(X_train, y_train, X_val, y_val, 'GMM Clustering of Combined Data', n_components_pca=2, n_components_gmm=3)
    plot_gmm(X_train, y_train, X_val, y_val, 'GMM Clustering of Combined Data', n_components_pca=3, n_components_gmm=3)
