import numpy as np
from fast_pytorch_kmeans import KMeans
from scipy.spatial import distance

from .data_helper import DataHelper


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
        np.random.seed(self.random_seed)

        self.model = KMeans(n_clusters=self.n_clusters, mode='cosine', verbose=0)

        embeddings = DataHelper.get_embeddings_as_torch(self.data_helper.df)

        self.labels = self.model.fit_predict(embeddings.float())

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


