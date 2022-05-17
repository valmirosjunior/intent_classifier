import torch
import numpy as np
from fast_pytorch_kmeans import KMeans


class ClusteringHelper:
    DEFAULT_RANDOM_SEED = 42

    def __init__(self, n_clusters, embeddings, random_seed=None):
        self.n_clusters = n_clusters
        self.embeddings = embeddings
        self.random_seed = random_seed or self.DEFAULT_RANDOM_SEED

        np.random.seed(ClusteringHelper.DEFAULT_RANDOM_SEED)
        self.model = KMeans(n_clusters=self.n_clusters, mode='cosine', verbose=0)
        self.labels = self.model.fit_predict(torch.from_numpy(self.embeddings).float())

    def get_centroids_as_numpy(self):
        return self.model.centroids.cpu().detach().numpy()

    def get_labels_as_numpy(self):
        return self.labels.numpy()
