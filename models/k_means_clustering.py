from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def main(data):

    # Keep labels
    labels = data["label"]

    # Features only
    X = data.drop(["label"], axis=1).to_numpy()

    # K-Means with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)

    print("Cluster distribution:", np.bincount(clusters))

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    plt.title('K-Means Clustering')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.subplot(1, 2, 2)
    label_map = {'left': 0, 'centre': 1, 'right': 2}
    y_numeric = labels.map(label_map)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='viridis')
    plt.title('Actual Labels')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.tight_layout()
    plt.show()
