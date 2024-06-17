import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import DataProcessor


def plot_pca(X_train, y_train, X_val, y_val, title, labels, n_components=2):
    X_combined = np.concatenate((X_train, X_val), axis=0)
    y_combined = np.concatenate((y_train, y_val), axis=0)

    X_reshaped = X_combined.reshape(X_combined.shape[0], -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    information_loss = 1 - np.sum(explained_variance_ratio)
    print(f"Amount of information lost after dimensionality reduction to {n_components}D:", information_loss)

    if n_components == 2:
        plt.figure(figsize=(8, 6))
        for class_label in np.unique(y_combined):
            plt.scatter(X_pca[y_combined == class_label, 0], X_pca[y_combined == class_label, 1],
                        label=labels[class_label])
        plt.title(title + ' - 2D')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Class')
        plt.show()

    elif n_components == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for class_label in np.unique(y_combined):
            ax.scatter(X_pca[y_combined == class_label, 0], X_pca[y_combined == class_label, 1],
                       X_pca[y_combined == class_label, 2], label=labels[class_label])
        ax.set_title(title + ' - 3D')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend(title='Class')
        plt.show()


if __name__ == '__main__':
    X_train, y_train, X_val, y_val = DataProcessor.train_test_data()
    labels = ['cat', 'dog', 'wild']

    plot_pca(X_train, y_train, X_val, y_val, 'PCA Projection of Combined Data', labels, n_components=2)
    plot_pca(X_train, y_train, X_val, y_val, 'PCA Projection of Combined Data', labels, n_components=3)
