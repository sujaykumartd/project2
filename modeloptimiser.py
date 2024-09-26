import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator

class clustering_model_optimizer:
    """
    A class for optimizing clustering models.

    This class provides methods to optimize the number of clusters for different
    clustering algorithms (K-means, Agglomerative, Spectral).

    Attributes:
        data (pd.DataFrame): The input data for clustering.
        min_clusters (int): The minimum number of clusters to consider.
        max_clusters (int): The maximum number of clusters to consider.
        best_N_cluster (int): The optimal number of clusters determined by the optimization.
    """

    def __init__(self, data):
        self.data = data
        self.min_clusters = 4
        self.max_clusters = 10
        print(f"Initialized clustering_model_optimizer with data shape: {self.data.shape}")
        
    def kmeans_n_cluster_optimizer(self):
        """
        Optimize the number of clusters for K-means clustering.

        This method uses the elbow method to determine the optimal number of clusters
        for K-means clustering.

        The result is stored in the best_N_cluster attribute.
        """
        print("Starting K-means optimization")
        wcss = []
        for i in range(self.min_clusters, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)
            print(f"K-means with {i} clusters: WCSS = {kmeans.inertia_}")
            
        kn = KneeLocator(range(self.min_clusters, self.max_clusters + 1), wcss, curve='convex', direction='decreasing')
        self.best_N_cluster = kn.knee
        print(f"K-means optimization complete. Best number of clusters: {self.best_N_cluster}")

    def agglomerative_spectural_n_cluster_optimizer(self, model_name):
        """
        Optimize the number of clusters for Agglomerative or Spectral clustering.

        This method evaluates different numbers of clusters using silhouette score
        and Davies-Bouldin score.

        Parameters:
        -----------
        model_name : str
            The name of the clustering model ('agglomerative' or 'spectral').

        The result is stored in the best_N_cluster attribute.
        """
        print(f"Starting {model_name} clustering optimization")
        score_dict = {'n_clusters': [], 'silhouette_score': [], 'DB_score': []}

        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            if model_name == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            elif model_name == 'spectral':
                model = SpectralClustering(n_clusters=n_clusters)
            else:
                raise ValueError('Unknown cluster model name')
                
            labels = model.fit_predict(self.data)
            silhouette_score_ = silhouette_score(self.data, labels)
            davies_bouldin_score_ = davies_bouldin_score(self.data, labels)
            score_dict['n_clusters'].append(n_clusters)
            score_dict['silhouette_score'].append(silhouette_score_)
            score_dict['DB_score'].append(davies_bouldin_score_)
            print(f"{model_name} with {n_clusters} clusters: Silhouette = {silhouette_score_:.4f}, Davies-Bouldin = {davies_bouldin_score_:.4f}")
        
        score_df = pd.DataFrame(score_dict).sort_values(['silhouette_score', 'DB_score'], ascending=[False, True])        
        self.best_N_cluster = score_df.n_clusters.iloc[0]
        print(f"{model_name} optimization complete. Best number of clusters: {self.best_N_cluster}")
        
    def agglomerative_n_cluster_optimizer(self):
        """
        Optimize the number of clusters for Agglomerative clustering.

        This method calls the agglomerative_spectural_n_cluster_optimizer method
        with the 'agglomerative' model name.
        """
        print("Starting Agglomerative clustering optimization")
        self.agglomerative_spectural_n_cluster_optimizer(model_name='agglomerative')

    def spectral_n_cluster_optimizer(self):
        """
        Optimize the number of clusters for Spectral clustering.

        This method calls the agglomerative_spectural_n_cluster_optimizer method
        with the 'spectral' model name.
        """
        print("Starting Spectral clustering optimization")
        self.agglomerative_spectural_n_cluster_optimizer(model_name='spectral')