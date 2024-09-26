import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap.umap_ as umap
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from modeloptimiser import clustering_model_optimizer
from Deminsionoptimiser import DimensionReductionOptimiser

class best_clustering_model_selector:
    """
    A class for selecting the best clustering model based on various preprocessing, 
    dimension reduction, and clustering techniques.

    This class provides methods for data preprocessing, feature engineering, 
    dimension reduction, clustering, and model evaluation.

    Attributes:
        labels (list): Cluster labels assigned to data points.
        best_n_cluster (int): The optimal number of clusters.
        data (pd.DataFrame): The input data for clustering.
        scalers (dict): Dictionary of scaling methods.
        dimension_reduction (dict): Dictionary of dimension reduction techniques.
        clusters (dict): Dictionary of clustering algorithms.
    """

    def __init__(self, df):
        """
        Initialize the best_clustering_model_selector.

        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe for clustering.
        """
        print("Initializing best_clustering_model_selector")
        self.labels = []
        self.best_n_cluster = None
        self.data = df
        self.scalers = {
            'normalization': MinMaxScaler(),
            'standardization': StandardScaler()
        }
        self.dimension_reduction = {
            'PCA': PCA(),
            'UMAP': umap.UMAP(n_jobs=-1),
            'TSNE': TSNE()
        }
        self.clusters = {
            'KMeans': KMeans(random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Agglomerative': AgglomerativeClustering(),
            'Spectral': SpectralClustering(random_state=42)
        }
        self.feature_engineering()
        print("Initialization complete")
        
    def prepare_data_for_preprocessing(self):
        """
        Prepare data for preprocessing by categorizing features.

        This method categorizes features into one-hot encoded, ordinal encoded, 
        and numerical features.
        """
        print("Preparing data for preprocessing")
        self.data.set_index('Customer ID', inplace=True)
        one_hot_cat_col = []
        ordinal_cat_col = []
        numerical_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.data.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            unique_values = self.data[feature].nunique()
            if unique_values < 4:
                one_hot_cat_col.append(feature)
            else:
                ordinal_cat_col.append(feature)
        self.one_hot_cat_col = one_hot_cat_col
        self.ordinal_cat_col = ordinal_cat_col
        self.numerical_col = numerical_features
        print(f"Identified {len(one_hot_cat_col)} one-hot columns, {len(ordinal_cat_col)} ordinal columns, and {len(numerical_features)} numerical columns")
        
    def data_preprocess_pipeline(self):
        """
        Create and apply the data preprocessing pipeline.

        This method creates pipelines for numerical and categorical data, 
        and applies them to the input data.
        """
        print("Starting data preprocessing pipeline")
        numerical_pipeline = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
        ])
        onehot_categorical_pipeline = Pipeline(steps=[
            ('onehot_encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
        ordinal_categorical_pipeline = Pipeline(steps=[
            ('ordinal_encoding', OrdinalEncoder())
        ])
        column_transform = ColumnTransformer([
            ('numerical_columns', numerical_pipeline, self.numerical_col),
            ('onehot_categorical_columns', onehot_categorical_pipeline, self.one_hot_cat_col),
            ('ordinal_categorical_columns', ordinal_categorical_pipeline, self.ordinal_cat_col)   
        ])
        self.trans_df = pd.DataFrame(column_transform.fit_transform(self.data)) 
        one_hot_cols = list(onehot_categorical_pipeline.fit(self.data[self.one_hot_cat_col]).get_feature_names_out())   
        trans_cols = list(self.numerical_col) + one_hot_cols + list(self.ordinal_cat_col)  
        self.trans_df.columns = trans_cols 
        print(f"Preprocessing complete. Transformed dataframe shape: {self.trans_df.shape}")
        
    def remove_constant_multicolinear_feature(self):
        """
        Remove constant and multicollinear features from the dataset.

        This method identifies and removes features with zero variance and 
        highly correlated features.
        """
        print("Removing constant and multicollinear features")
        initial_shape = self.trans_df.shape
        std_df = self.trans_df.describe().T['std']
        const_feature_list = list(std_df[std_df == 0].index)
        if len(const_feature_list) > 0:
            self.trans_df.drop(const_feature_list, axis=1, inplace=True)

        corr_mat = self.trans_df.corr().abs()
        upper_cor_mat_df = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        drop_col = [col for col in upper_cor_mat_df.columns if any(upper_cor_mat_df[col] > 0.85)]

        self.data.reset_index(inplace=True)
        self.trans_df = pd.concat([self.data['Customer ID'], self.trans_df], axis=1)
        self.trans_df.drop(drop_col, axis=1, inplace=True)
        print(f"Removed {initial_shape[1] - self.trans_df.shape[1]} features. New shape: {self.trans_df.shape}")
        
    def feature_engineering(self):
        """
        Perform feature engineering on the input data.

        This method calls the data preparation, preprocessing, and feature 
        removal methods in sequence.
        """
        print("Starting feature engineering process")
        self.prepare_data_for_preprocessing()
        self.data_preprocess_pipeline()
        self.remove_constant_multicolinear_feature()
        print("Feature engineering complete")
        
    def scaling_data(self, scaling_method):
        """
        Scale the data using the specified scaling method.

        Parameters:
        -----------
        scaling_method : str
            The scaling method to use ('normalization' or 'standardization').

        Raises:
        -------
        ValueError
            If an unknown scaling method is provided.
        """
        print(f"Scaling data using {scaling_method} method")
        if scaling_method in self.scalers.keys():
            scaling_pipeline = Pipeline(steps=[('scaler', self.scalers[scaling_method])])
            temp_df = self.trans_df.drop(self.numerical_col, axis=1)
            column_transform = ColumnTransformer([('numerical_columns', scaling_pipeline, self.numerical_col)])
            self.scaled_df = pd.DataFrame(column_transform.fit_transform(self.trans_df))
            self.scaled_df.columns = self.numerical_col
            self.scaled_df = pd.concat([temp_df, self.scaled_df], axis=1)
            self.scaled_df.set_index('Customer ID', inplace=True)
            print(f"Scaling complete. Scaled dataframe shape: {self.scaled_df.shape}")
        else:
            raise ValueError("Unknown Data Scaling method")
    
    def reduce_dimensions(self, dimension_method='PCA', scaling_method='normalization'):
        """
        Reduce the dimensionality of the data using the specified method.

        Parameters:
        -----------
        dimension_method : str, optional
            The dimension reduction method to use (default is 'PCA').
        scaling_method : str, optional
            The scaling method used for the data (default is 'normalization').

        Raises:
        -------
        ValueError
            If an unknown dimensionality reduction method is provided.
        """
        print(f"Reducing dimensions using {dimension_method} method")
        if dimension_method in self.dimension_reduction.keys():
            self.dim_red_n_component = self.reduce_dim_n_component_dict[scaling_method][dimension_method]
            self.reducer = self.dimension_reduction[dimension_method]
            self.reducer.n_components = self.dim_red_n_component
        else:
            raise ValueError("Unknown dimensionality reduction method")
        
        self.reduced_df = self.reducer.fit_transform(self.scaling_data_dict[scaling_method])
        print(f"Dimension reduction complete. Reduced dataframe shape: {self.reduced_df.shape}")
    
    def reduce_data_parallel(self, scaling_methods, dim_reduction_methods):
        """
        Perform parallel dimension reduction for multiple scaling and reduction methods.

        Parameters:
        -----------
        scaling_methods : list
            List of scaling methods to use.
        dim_reduction_methods : list
            List of dimension reduction methods to use.
        """
        print("Starting parallel data reduction")
        scaling_data_dict = {scaling: [] for scaling in scaling_methods}
        
        for scaling in scaling_methods:
            self.scaling_data(scaling)
            scaling_data_dict[scaling] = self.scaled_df
        self.scaling_data_dict = scaling_data_dict
        optimiser = DimensionReductionOptimiser(self.scaling_data_dict, scaling_methods, dim_reduction_methods)
        optimiser.optimize_all()
        self.reduce_dim_n_component_dict = optimiser.n_components_dict
        print("Parallel data reduction complete")
    
    def clustering_data(self, cluster_method='KMeans'):
        """
        Perform clustering on the reduced data using the specified method.

        Parameters:
        -----------
        cluster_method : str, optional
            The clustering method to use (default is 'KMeans').

        Returns:
        --------
        list
            The cluster labels assigned to each data point.

        Raises:
        -------
        ValueError
            If an unknown clustering method is provided.
        """
        print(f"Clustering data using {cluster_method} method")
        if cluster_method in self.clusters.keys():
            if cluster_method != 'DBSCAN':
                n_cluster_selector = clustering_model_optimizer(self.reduced_df)
                method_name = f'{cluster_method.lower()}_n_cluster_optimizer'
                method = getattr(n_cluster_selector, method_name, None)
                if method:
                    method()
                    self.best_n_cluster = n_cluster_selector.best_N_cluster
                    self.clusterer = self.clusters[cluster_method]
                    if self.best_n_cluster is not None: 
                        self.clusterer.n_clusters = self.best_n_cluster
                        self.labels = self.clusterer.fit_predict(self.reduced_df)
                        print(f"Clustering complete. Number of clusters: {self.best_n_cluster}")
                    else:
                        print(f'{cluster_method} model is not built')
                else:
                    self.labels = []
                    raise ValueError(f"{method} not found")
            elif cluster_method == 'DBSCAN':
                self.clusterer = self.clusters[cluster_method]
                self.labels = self.clusterer.fit_predict(self.reduced_df)
                print(f"DBSCAN clustering complete. Number of clusters: {len(set(self.labels))}")
        else:
            self.labels = []
            raise ValueError("Unknown clustering method")
        return self.labels
    
    def evaluate_clustering(self):
        """
        Evaluate the clustering results using silhouette and Davies-Bouldin scores.

        Returns:
        --------
        tuple
            A tuple containing the silhouette score and Davies-Bouldin score.
        """
        print("Evaluating clustering results")
        silhouette = silhouette_score(self.scaled_df, self.labels)
        davies_bouldin = davies_bouldin_score(self.scaled_df, self.labels)
        print(f"Evaluation complete. Silhouette score: {silhouette:.4f}, Davies-Bouldin score: {davies_bouldin:.4f}")
        return silhouette, davies_bouldin
    
    def clustering_worker(self, params):
        """
        Worker function for parallel clustering execution.

        Parameters:
        -----------
        params : tuple
            A tuple containing scaling method, dimension reduction method, 
            and clustering method.

        Returns:
        --------
        tuple or None
            A tuple containing clustering results and scores, or None if clustering failed.
        """
        scaling_method, dim_red_method, cluster_method = params
        print(f"Starting clustering worker with {scaling_method}, {dim_red_method}, {cluster_method}")
        self.scaling_data(scaling_method)
        self.reduce_dimensions(dimension_method=dim_red_method, scaling_method=scaling_method)
        
        self.clustering_data(cluster_method=cluster_method)
        
        if len(set(self.labels)) > 1:
            silhouette, davies_bouldin = self.evaluate_clustering()
            print(f"Worker complete. Silhouette: {silhouette:.4f}, Davies-Bouldin: {davies_bouldin:.4f}")
            return (scaling_method, dim_red_method, cluster_method, silhouette, davies_bouldin, self.best_n_cluster)
        else:
            print("Worker failed: Only one cluster found")
            return None
            
    def compare_models_parallel(self, scaling_methods, dim_reduction_methods, clustering_methods):
        """
        Compare multiple clustering models in parallel.

        This method runs different combinations of scaling, dimension reduction, 
        and clustering methods in parallel and collects the results.

        Parameters:
        -----------
        scaling_methods : list
            List of scaling methods to use.
        dim_reduction_methods : list
            List of dimension reduction methods to use.
        clustering_methods : list
            List of clustering methods to use.
        """
        print("Starting parallel model comparison")
        score_dict = {  
            'norma_method': [],
            'dim_red_method': [],
            'dim_red_n_component': [],
            'clustering_method': [],
            'clustering_n_clusters': [],
            'silhouette_score': [],
            'DB_score': []
        }

        param_combinations = [(scaling_method, dim_red_method, cluster_method)
                              for cluster_method in clustering_methods
                              for scaling_method in scaling_methods
                              for dim_red_method in dim_reduction_methods]
        
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.clustering_worker, i): i for i in param_combinations}
            for future in as_completed(futures):
                method = futures[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        scaling_method, dim_red_method, cluster_method, silhouette, davies_bouldin, best_n_cluster = result
                        score_dict['norma_method'].append(scaling_method)
                        score_dict['dim_red_method'].append(dim_red_method)
                        score_dict['dim_red_n_component'].append(self.reduce_dim_n_component_dict[scaling_method][dim_red_method])
                        score_dict['clustering_method'].append(cluster_method)
                        score_dict['clustering_n_clusters'].append(best_n_cluster)
                        score_dict['silhouette_score'].append(silhouette)
                        score_dict['DB_score'].append(davies_bouldin)
                        print(f"Added results for {scaling_method}, {dim_red_method}, {cluster_method}")
                except Exception as exc:
                    print(f'{method} generated an exception: {exc}')
    
        self.score_dict = score_dict 
        print("Parallel model comparison complete")
    
    def plot_clusters(self, X_reduced, labels, title='Cluster Plot'):
        """
        Plot the clustering results.

        Parameters:
        -----------
        X_reduced : array-like
            The reduced-dimensional data to plot.
        labels : array-like
            The cluster labels for each data point.
        title : str, optional
            The title of the plot (default is 'Cluster Plot').
        """
        print("Plotting clusters")
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, s=50, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
        print("Cluster plot displayed")