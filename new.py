import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, make_scorer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.neighbors import NearestNeighbors
import hdbscan
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.stats.outliers_influence import variance_inflation_factor
from kneed import KneeLocator

class ModelSelector:
    def __init__(self, data):
        self.data = data
        self.transformed_data = None
        self.scaled_data = {}
        self.reduced_data = {}
        self.results = []
        self.preprocessor = None
        self.random_seed = 42
        print("ModelSelector initialized")

    def prepare_data_for_preprocessing(self):
        print("Starting prepare_data_for_preprocessing")
        # Remove features with 80% or more missing values
        missing_threshold = len(self.data) * 0.8
        filtered_data = self.data.dropna(axis=1, thresh=missing_threshold)
        
        # Identify numeric and categorical columns  
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
        print("Finished prepare_data_for_preprocessing")

    def data_preprocess_pipeline(self):
        print("Starting data_preprocess_pipeline")
        # Build preprocessing pipeline for numerical features
        print("** KNN imputer for missing values")
        numerical_pipeline = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
        #     ('scaler', MinMaxScaler())
        ])
        # Build preprocessing pipeline for categorical features
        print("** One hot encoding for the Categorical features with less than 4 unique values")
        
        onehot_categorical_pipeline = Pipeline(steps=[
            ('onehot_encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
        
        print("** Ordinal encoding for the Categorical features with greater than 4 unique values")

        ordinal_categorical_pipeline = Pipeline(steps=[
            ('ordinal_encoding', OrdinalEncoder())

        ])
        column_transform = ColumnTransformer([
            ('numerical_columns', numerical_pipeline, self.numerical_col),
            ('onehot_categorical_columns', onehot_categorical_pipeline, self.one_hot_cat_col),
            ('ordinal_categorical_columns', ordinal_categorical_pipeline, self.ordinal_cat_col)   
        ])
        self.transformed_data = pd.DataFrame(column_transform.fit_transform(self.data)) 
        one_hot_cols = list(onehot_categorical_pipeline.fit(self.data[self.one_hot_cat_col]).get_feature_names_out())   
        trans_cols = list(self.numerical_col)+one_hot_cols+list(self.ordinal_cat_col)  
        self.transformed_data.columns = trans_cols 
        self.transformed_data.to_csv('transformed_data.csv', index=False)
        print("Finished data_preprocess_pipeline")

    def remove_multicollinearity(self, correlation_threshold=0.85, use_vif=False, vif_threshold=5):
        print("Starting remove_multicollinearity")
        # Remove highly correlated features
        corr_matrix = self.transformed_data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        self.transformed_data = self.transformed_data.drop(to_drop, axis=1)

        if use_vif:
            # Calculate VIF and remove features with high VIF
            X = self.transformed_data.values
            vif = pd.DataFrame()
            vif["features"] = self.transformed_data.columns
            vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            
            while vif["VIF"].max() > vif_threshold:
                feature_to_drop = vif.loc[vif["VIF"].idxmax(), "features"]
                self.transformed_data = self.transformed_data.drop(feature_to_drop, axis=1)
                
                X = self.transformed_data.values
                vif = pd.DataFrame()
                vif["features"] = self.transformed_data.columns
                vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        print("Finished remove_multicollinearity")

    def scale_data(self):
        print("Starting scale_data")
        self.scaled_data['normalization'] = MinMaxScaler().fit_transform(self.transformed_data)
        self.scaled_data['standardization'] = StandardScaler().fit_transform(self.transformed_data)
        print("Finished scale_data")

    def find_best_n_components(self, method, threshold=0.95):
        print(f"Starting find_best_n_components for {method}")
        if method == 'PCA':
            pca = PCA(random_state=self.random_seed)
            pca.fit(self.scaled_data['standardization'])  # Use standardized data for PCA
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= threshold) + 1
            result = max(4, min(n_components, 10))  # Ensure it's between 4 and 10
        elif method == 'UMAP':
            n_components_range = range(4, 11)
            with multiprocessing.Pool() as pool:
                evaluate_func = partial(self.evaluate_n_components, method=method)
                results = pool.map(evaluate_func, n_components_range)
            result, _ = max(zip(n_components_range, results), key=lambda x: x[1])
        elif method == 'TSNE':
            n_components_range = [2, 3]
            with multiprocessing.Pool() as pool:
                evaluate_func = partial(self.evaluate_n_components, method=method)
                results = pool.map(evaluate_func, n_components_range)
            result, _ = max(zip(n_components_range, results), key=lambda x: x[1])
        else:
            raise ValueError("Unsupported dimension reduction method")
        print(f"Finished find_best_n_components for {method}. Best n_components: {result}")
        return result

    def evaluate_n_components(self, n, method):
        print(f"Evaluating {n} components for {method}")
        reduced_data = self.apply_dimension_reduction(method, n, 'standardization')
        kmeans = KMeans(n_clusters=3, random_state=self.random_seed)
        labels = kmeans.fit_predict(reduced_data)
        score = silhouette_score(self.scaled_data['standardization'], labels)
        print(f"Silhouette score for {n} components: {score}")
        return score

    def apply_dimension_reduction(self, method, n_components, scaling_method):
        print(f"Applying {method} with {n_components} components")
        if method == 'PCA':
            reducer = PCA(n_components=n_components, random_state=self.random_seed)
        elif method == 'TSNE':
            reducer = TSNE(n_components=n_components, random_state=self.random_seed)
        elif method == 'UMAP':
            reducer = UMAP(n_components=n_components, random_state=self.random_seed)
        result = reducer.fit_transform(self.scaled_data[scaling_method])
        print(f"Finished applying {method}")
        return result

    def evaluate_model(self, data, labels):
        print("Evaluating model")
        sil_score = silhouette_score(data, labels)
        db_score = davies_bouldin_score(data, labels)
        print(f"Silhouette score: {sil_score}, Davies-Bouldin score: {db_score}")
        return sil_score, db_score

    def process_model(self, args):
        scaling_method, dim_method, model_name, model = args
        print(f"Processing model: {model_name} with {scaling_method} scaling and {dim_method} dimension reduction")
        reduced_data = self.reduced_data[(scaling_method, dim_method)]
        scaled_data = self.scaled_data[scaling_method]

        n_clusters = None  # Default for DBSCAN and HDBSCAN
        if model_name == 'KMeans' or model_name == 'Agglomerative':
            n_clusters = self.select_best_n_clusters(reduced_data, model_name)
            if n_clusters is not None:
                    model.set_params(n_clusters=n_clusters)
                    labels = model.fit_predict(reduced_data)
            else:
                print(f"Model not created because n_clusters is None for {model_name}")
                return None 

        if model_name == 'HDBSCAN':
            labels = model.fit_predict(reduced_data)
        else:
            labels = model.fit_predict(reduced_data)

        if len(np.unique(labels)) < 4:
            print(f"Model not created because less than 4 clusters were found for {model_name}")
            return None
        sil_score, db_score = self.evaluate_model(scaled_data, labels)
        result = {
            'model': model_name,
            'scaling_method': scaling_method,
            'dim_reduction_method': dim_method,
            'n_components': self.n_components[dim_method],
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'davies_bouldin_score': db_score
        }
        print(f"Finished processing model: {model_name}")
        return result

    def select_best_n_clusters(self, data, model_type):
        print(f"Selecting best n_clusters for {model_type}")
        cluster_range = range(4, 11)

        if model_type == 'KMeans':
            # KMeans - Elbow Method
            inertias = []
            for n in cluster_range:
                kmeans = KMeans(n_clusters=n, random_state=self.random_seed)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)

            kl = KneeLocator(cluster_range, inertias, curve='convex', direction='decreasing')
            result = kl.elbow
        elif model_type == 'Agglomerative':
            # Agglomerative - Using evaluate_model function
            scores = []
            for n in cluster_range:
                agglomerative = AgglomerativeClustering(n_clusters=n)
                labels = agglomerative.fit_predict(data)
                sil_score, _ = self.evaluate_model(data, labels)
                scores.append(sil_score)

            result = cluster_range[np.argmax(scores)]
        print(f"Best n_clusters for {model_type}: {result}")
        return result

    def select_best_model(self, correlation_threshold=0.85, use_vif=False, vif_threshold=5):
        print("Starting select_best_model")
        self.prepare_data_for_preprocessing()
        self.data_preprocess_pipeline()
        self.remove_multicollinearity(correlation_threshold, use_vif, vif_threshold)
        self.scale_data()
        scaling_methods = ['normalization', 'standardization']
        dim_reduction_methods = ['PCA', 'TSNE', 'UMAP']

        models = {
            'KMeans': KMeans(random_state=self.random_seed),  # n_clusters will be set in process_model
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Agglomerative': AgglomerativeClustering(),  # n_clusters will be set in process_model
            'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=5)
        }

        # Find best n_components for each dimension reduction method
        self.n_components = {method: self.find_best_n_components(method) for method in dim_reduction_methods}

        # Apply dimension reduction once for each combination of scaling and dimension reduction method
        for scaling in scaling_methods:
            for dim_method in dim_reduction_methods:
                self.reduced_data[(scaling, dim_method)] = self.apply_dimension_reduction(
                    dim_method, self.n_components[dim_method], scaling
                )

        tasks = [(scaling, dim, model_name, model) 
                 for scaling in scaling_methods 
                 for dim in dim_reduction_methods 
                 for model_name, model in models.items()]

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_model, task) for task in tasks]
            for future in as_completed(futures):
                temp_result = future.result()
                if temp_result is not None:
                    self.results.append(temp_result)

        # Sort results by silhouette score (descending) and Davies-Bouldin score (ascending)
        sorted_results = sorted(self.results, key=lambda x: (x['silhouette_score'], -x['davies_bouldin_score']), reverse=True)
        best_model = sorted_results[0]

        # Store sorted results as DataFrame
        results_df = pd.DataFrame(sorted_results)

        # Return the scaling method and dimension reduction method used in the best model
        best_scaling = best_model['scaling_method']
        best_dim_reduction = best_model['dim_reduction_method']
        best_n_components = best_model['n_components']
        best_n_clusters = best_model['n_clusters']

        print("Finished select_best_model")
        return best_model, best_scaling, best_dim_reduction, best_n_components, best_n_clusters, results_df

def load_data(file_path):
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    print("Data loaded successfully")
    return data


# Hyperparameter Tuning Class
class HyperparameterTuner:
    def __init__(self, best_model, scaled_data, reduced_data, n_clusters, random_seed=42):
        self.best_model = best_model
        self.scaled_data = scaled_data
        self.reduced_data = reduced_data
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.final_model = None
        self.best_params = None
        print("HyperparameterTuner initialized")

    def custom_silhouette_scorer(self, estimator, X):
        # print("Calculating custom silhouette score")
        labels = estimator.fit_predict(self.scaled_data)
        if len(set(labels)) > 1:  # Ensure we have more than one cluster
            score = silhouette_score(self.scaled_data, labels)
        else:
            score = -1  # Penalize single cluster solutions
        # print(f"Custom silhouette score: {score}")
        return score

    def tune_hyperparameters(self):
        print("Starting hyperparameter tuning")
        clustering_method = self.best_model

        # Create the custom scorer
        custom_scorer = make_scorer(self.custom_silhouette_scorer, greater_is_better=True)

        # Unified tuning function
        self.final_model, self.best_params = self.tune_model(self.scaled_data, clustering_method, self.n_clusters, custom_scorer)
        print("Finished hyperparameter tuning")
        return self.final_model, self.best_params

    def tune_model(self, data, clustering_method, n_clusters, custom_scorer):
        print(f"Tuning model for {clustering_method}")
        if clustering_method == 'KMeans':
            model = KMeans(random_state=self.random_seed)
            param_grid = {
                'n_clusters': [n_clusters],
                'init': ['k-means++', 'random'],
                'n_init': [10, 20, 30],
                'max_iter': [200, 300, 500]
            }
        elif clustering_method == 'Agglomerative':
            model = AgglomerativeClustering()
            param_grid = {
                'n_clusters': [n_clusters],
                'linkage': ['ward', 'complete', 'average'],
            }
            if 'ward' in param_grid['linkage']:
                param_grid['metric'] = ['euclidean']
            else:
                param_grid['metric'] = ['euclidean', 'manhattan', 'cosine']
        elif clustering_method == 'DBSCAN':
            # Determine optimal eps using k-distance graph
            neighbors = NearestNeighbors(n_neighbors=2)
            nbrs = neighbors.fit(data)
            distances, _ = nbrs.kneighbors(data)
            distances = np.sort(distances, axis=0)
            distances = distances[:,1]
            
            knee = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
            optimal_eps = distances[knee.knee]

            model = DBSCAN()
            param_grid = {
                'eps': [optimal_eps, optimal_eps*0.8, optimal_eps*1.2],
                'min_samples': range(2, 11)
            }
        elif clustering_method == 'HDBSCAN':
            model = hdbscan.HDBSCAN()
            param_grid = {
                'min_cluster_size': range(5, 51, 5),
                'min_samples': range(1, 11),
                'cluster_selection_epsilon': [0, 0.1, 0.5, 1.0],
                'alpha': [0.5, 1.0, 1.5]
            }
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")

        if clustering_method in ['KMeans', 'Agglomerative', 'DBSCAN']:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring=custom_scorer, n_jobs=-1)
            grid_search.fit(data)
            print(f"Best parameters for {clustering_method}: {grid_search.best_params_}")
            return grid_search.best_estimator_, grid_search.best_params_
        elif clustering_method == 'HDBSCAN':
            best_score = -np.inf
            best_params = None
            best_model = None

            for params in ParameterGrid(param_grid):
                hdbscan_model = hdbscan.HDBSCAN(**params)
                labels = hdbscan_model.fit_predict(data)
                if len(np.unique(labels)) < 4:
                    continue
                score = self.custom_silhouette_scorer(hdbscan_model, data)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = hdbscan_model

            if best_model is None:
                print("No valid HDBSCAN model found with at least 4 clusters")
                return None, None

            print(f"Best parameters for HDBSCAN: {best_params}")
            return best_model, best_params

    def generate_final_model_and_predict(self, original_data):
        print("Generating final model and predictions")
        if self.final_model is None or self.best_params is None:
            raise ValueError("Please run tune_hyperparameters() first.")

        # Generate predictions
        predictions = self.final_model.fit_predict(self.scaled_data)

        # Add predictions to original data
        result_data = original_data.copy()
        result_data['Cluster'] = predictions

        # Save results as CSV
        result_data.to_csv('clustering_results.csv', index=False)

        print("Clustering results have been saved to 'clustering_results.csv'")
        return result_data

def run_clustering_pipeline(data_path):
    print("Starting clustering pipeline")
    # Load the data
    data = load_data(data_path)
    
    # Create ModelSelector object and select the best model
    selector = ModelSelector(data)
    best_model, best_scaling, best_dim_reduction, best_n_components, best_n_clusters, results_df = selector.select_best_model(use_vif=False)
    
    print("Best Model:", best_model)
    print("Best Scaling Method:", best_scaling)
    print("Best Dimension Reduction Method:", best_dim_reduction)
    print("Best Number of Components:", best_n_components)
    print("Best Number of Clusters:", best_n_clusters)
    print("\nAll Results:")
    print(results_df)

    # Prepare scaled and reduced data
    scaled_data = selector.scaled_data[best_scaling]
    reduced_data = selector.reduced_data[(best_scaling, best_dim_reduction)]

    # Use the HyperparameterTuner
    print("Starting hyperparameter tuning")

    tuner = HyperparameterTuner(best_model['model'], scaled_data, reduced_data, best_n_clusters)
    final_model, best_params = tuner.tune_hyperparameters()

    print("\nBest Hyperparameters:", best_params)
    print("Final Model:", final_model)

    # Generate final model, predict, and save results
    results_with_clusters = tuner.generate_final_model_and_predict(data)
    print("\nClustering completed. Results added to original data and saved as CSV.")

    return results_with_clusters

if __name__ == "__main__":
    results = run_clustering_pipeline('data.csv')
    print("Clustering pipeline completed successfully.")
