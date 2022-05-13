import numpy as np
from fast_pytorch_kmeans import KMeans
from scipy.spatial import distance

from .data_helper import DataHelper


def clustering(k, X):
    np.random.seed(ClusteringHelper.DEFAULT_RANDOM_SEED)

    model = KMeans(
        n_clusters=k, mode='cosine', verbose=0
    )

    labels = model.fit_predict(X)

    return labels, model


class ClusteringHelper:
    DEFAULT_RANDOM_SEED = 42

    def __init__(self, n_clusters, embedding_name, actor, data_helper: DataHelper, random_seed=None):
        self.data_helper = data_helper
        self.actor = actor
        self.embedding_name = embedding_name
        self.n_clusters = n_clusters
        self.random_seed = random_seed or self.DEFAULT_RANDOM_SEED

        self.clustering_data()

    def clustering_data(self):
        embeddings = DataHelper.get_embeddings_as_torch(self.data_helper.df)

        self.labels, self.model = clustering(self.n_clusters, embeddings)

    def get_centroids(self):
        return self.model.centroids.cpu().detach().numpy()

    def get_labels(self):
        return self.labels.numpy()

    def get_distance_from_centroid(self):
        centroids = self.get_centroids()

        return self.data_helper.df.apply(
            lambda row: distance.cosine(row['embeddings'], centroids[row['label']]), axis=1
        )

    def get_distances(self, label):
        return self.data_helper.df[self.data_helper.df['label'] == label]['distance'].to_numpy()


