# No external libraries are allowed to be imported in this file
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# Function to generate the dataset
def generate_blobs_dataset(n_samples, centers, n_features, random_state):
    """
    Generates a 3D dataset using make_blobs.

    Parameters:
    n_samples (int): Number of samples.
    centers (int): Number of centers.
    n_features (int): Number of features (3 for 3D data).
    random_state (int): Seed for reproducibility.

    Returns:
    tuple: Generated data (X, y)
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
    )
    return X, y


# Function to plot the original 3D data
def plot_3d_data(X, y):
    # TO DO: Use scatter plot to visualize the original data in 3D space
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis")
    ax.set_title("3D")
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.set_zlabel("F3")
    plt.show()


# Function to standardize the dataset
def standardize_data(X):
    """
    Standardizes the dataset using StandardScaler.

    Parameters:
    X (array): Original dataset.

    Returns:
    array: Scaled dataset.
    """
    # TO DO: Instantiate StandardScaler and use fit_transform to scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# Function to apply PCA to the dataset
def apply_pca(X_scaled, n_components):
    """
    Applies PCA to reduce dimensionality of the dataset.

    Parameters:
    X_scaled (array): Scaled dataset.
    n_components (int): Number of principal components.

    Returns:
    array: PCA transformed data.
    PCA object: The PCA object used.
    """
    # TO DO: Instantiate PCA with n_components and apply PCA transformation
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


# Function to plot the 2D PCA projection
def plot_pca_projection(X_pca, y):
    # TO DO: Use scatter plot to visualize the 2D projection from PCA
    ax = plt.figure().add_subplot()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
    ax.set_title("2D (PCA)")
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    plt.show()


if __name__ == "__main__":
    np.random.seed(2024)
    X, y = generate_blobs_dataset(
        n_samples=300, centers=2, n_features=3, random_state=2024
    )
    plot_3d_data(X, y)  # Visualize the original 3D dataset
    X_scaled = standardize_data(X)  # Standardize the dataset
    # plot_3d_data(X_scaled, y)  # Visualize the scaled 3D dataset

    # TO DO: Fill in appropriate value for n_components
    X_pca, pca = apply_pca(X_scaled, n_components=2)  # Apply PCA
    plot_pca_projection(X_pca, y)  # Visualize the 2D PCA projection
