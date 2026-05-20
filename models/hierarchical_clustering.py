import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

def main(data):

    labels = data["label"]
    features = data.drop(["label"], axis=1).to_numpy()

    # K-Means with 3 clusters
    agglomerative = AgglomerativeClustering()
    clusters = agglomerative.fit_predict(features)

    print("Cluster distribution:", np.bincount(clusters))

    # PCA
    pca = PCA(n_components=4)
    features_pca = pca.fit_transform(features)

    report = {
        "ari": adjusted_rand_score(labels, agglomerative.labels_),
        "sil": silhouette_score(features, agglomerative.labels_),
    }

    df = pd.DataFrame.from_dict(report, orient='index')
    df.to_csv('evaluation/agglomerative_report.csv')

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis')
    plt.title('Agglomerative Clustering')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.subplot(1, 2, 2)
    label_map = {'left': 0, 'centre': 1, 'right': 2}
    y_numeric = labels.map(label_map)

    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=y_numeric, cmap='viridis')
    plt.title('Actual Labels')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.tight_layout()
    plt.show()
