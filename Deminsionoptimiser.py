import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator
import umap.umap_ as umap
from concurrent.futures import ProcessPoolExecutor, as_completed

class DimensionReductionOptimiser:
    """
    A class for optimizing dimension reduction techniques.

    This class provides methods to optimize the number of components for different
    dimension reduction techniques (PCA, t-SNE, UMAP) across various scaling methods.

    Attributes:
        data_dict (dict): Dictionary containing scaled data for different scaling methods.
        scaling_methods (list): List of scaling methods used.
        dim_reduction_methods (list): List of dimension reduction methods to optimize.
        pca_threshold (float): Threshold for cumulative explained variance in PCA.
        tsne_min_components (int): Minimum number of components for t-SNE.
        tsne_max_components (int): Maximum number of components for t-SNE.
        umap_min_components (int): Minimum number of components for UMAP.
        umap_max_components (int): Maximum number of components for UMAP.
        n_components_dict (dict): Dictionary to store optimized number of components.
    """

    def __init__(self, data, scaling_methods, dim_reduction_methods):
        print("Initializing DimensionReductionOptimiser")
        self.data_dict = data
        self.scaling_methods = scaling_methods
        self.dim_reduction_methods = dim_reduction_methods
        self.pca_threshold = 0.95
        self.tsne_min_components = 2
        self.tsne_max_components = 3
        self.umap_min_components = 2
        self.umap_max_components = 6
        self.n_components_dict = {scale: {dim: None} for dim in dim_reduction_methods for scale in scaling_methods}
        print(f"Initialization complete. Scaling methods: {scaling_methods}, Dimension reduction methods: {dim_reduction_methods}")
          
    def pca_n_components_optimizer(self, scaling_method):
        """
        Optimize the number of components for PCA.

        This method determines the optimal number of principal components to retain
        based on the cumulative explained variance ratio.

        Parameters:
        -----------
        scaling_method : str
            The scaling method used for the input data.

        Returns:
        --------
        int
            The optimal number of principal components.
        """
        print(f"Starting PCA optimization for {scaling_method} scaling")
        pca = PCA()
        X_pca = pca.fit_transform(self.data_dict[scaling_method])
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        best_n_components = len(cumulative_variance[cumulative_variance <= self.pca_threshold])
        print(f"PCA optimization complete. Best number of components: {best_n_components}")
        return best_n_components

    def tsne_n_components_optimizer(self, scaling_method):
        """
        Optimize the number of components for t-SNE.

        This method determines the optimal number of components for t-SNE
        by evaluating the silhouette score for different numbers of components.

        Parameters:
        -----------
        scaling_method : str
            The scaling method used for the input data.

        Returns:
        --------
        int or None
            The optimal number of components for t-SNE, or None if no valid result is found.
        """
        print(f"Starting t-SNE optimization for {scaling_method} scaling")
        best_n_components = None
        best_score = -1

        for n_components in range(self.tsne_min_components, self.tsne_max_components + 1):
            print(f"Trying t-SNE with {n_components} components")
            tsne = TSNE(n_components=n_components)
            transformed_data = tsne.fit_transform(self.data_dict[scaling_method])
            kmeans = KMeans(n_clusters=4)
            labels = kmeans.fit_predict(transformed_data)
            
            if len(set(labels)) > 1:
                score = silhouette_score(transformed_data, labels)
                print(f"Silhouette score for {n_components} components: {score}")
                if score > best_score:
                    best_score = score
                    best_n_components = n_components

        print(f"t-SNE optimization complete. Best number of components: {best_n_components}")
        return best_n_components

    def umap_n_components_optimizer(self, scaling_method):
        """
        Optimize the number of components for UMAP.

        This method determines the optimal number of components for UMAP
        by evaluating the silhouette score for different numbers of components.

        Parameters:
        -----------
        scaling_method : str
            The scaling method used for the input data.

        Returns:
        --------
        int or None
            The optimal number of components for UMAP, or None if no valid result is found.
        """
        print(f"Starting UMAP optimization for {scaling_method} scaling")
        best_n_components = None
        best_score = -1

        for n_components in range(self.umap_min_components, self.umap_max_components + 1):
            print(f"Trying UMAP with {n_components} components")
            reducer = umap.UMAP(n_components=n_components)
            reduced_data = reducer.fit_transform(self.data_dict[scaling_method])
            kmeans = KMeans(n_clusters=4)
            labels = kmeans.fit_predict(reduced_data)
            
            if len(set(labels)) > 1:
                score = silhouette_score(reduced_data, labels)
                print(f"Silhouette score for {n_components} components: {score}")
                if score > best_score:
                    best_score = score
                    best_n_components = n_components

        print(f"UMAP optimization complete. Best number of components: {best_n_components}")
        return best_n_components
    
    def optimize_all(self):
        """
        Optimize the number of components for all dimension reduction methods and scaling methods.

        This method uses parallel processing to optimize the number of components
        for each combination of dimension reduction method and scaling method.

        The results are stored in the n_components_dict attribute.
        """
        print("Starting optimization for all methods")
        func_list = [(scale, dim) for dim in self.dim_reduction_methods for scale in self.scaling_methods]
        
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(getattr(self, f'{dim.lower()}_n_components_optimizer'), scaling): (scaling, dim) for scaling, dim in func_list}
            
            for future in as_completed(futures):
                method = futures[future]
                try:
                    result = future.result()
                    self.n_components_dict[method[0]][method[1]] = result
                    print(f"Optimization complete for {method[1]} with {method[0]} scaling. Result: {result}")
                except Exception as exc:
                    print(f'{method} generated an exception: {exc}')

        print("All optimizations complete")
        print("Final results:", self.n_components_dict)
