import pandas as pd
from model_selector import best_clustering_model_selector
import multiprocessing
import time
import logging
import os
from datetime import datetime

def setup_logging():
    log_dir = 'logging'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def log_and_print(message): 
    print(message)  # Keep print statement for debugging
    logging.info(message)

def main(): 
    setup_logging()
    log_and_print("Starting main function")
    start_time = time.time()

    # Load the data
    print("Loading data")  # Keep original print statement
    df = pd.read_csv('data.csv')  # Replace with your actual data file
    print(f"Data loaded, shape: {df.shape}")  # Keep original print statement
    log_and_print(f"Data loaded, shape: {df.shape}")

    # Initialize the model selector
    print("Initializing model selector")  # Keep original print statement
    select_obj = best_clustering_model_selector(df)

    # Define methods to use
    scaling_methods = ['normalization', 'standardization']
    dim_reduction_methods = ['PCA', 'UMAP', 'TSNE']
    clustering_methods = ['KMeans', 'DBSCAN', 'Agglomerative', 'Hdbscan']
    log_and_print(f"Methods defined: scaling={scaling_methods}, dim_reduction={dim_reduction_methods}, clustering={clustering_methods}")

    # Perform parallel dimension reduction
    print("Starting parallel dimension reduction")  # Keep original print statement
    select_obj.reduce_data_parallel(scaling_methods, dim_reduction_methods)
    print("Parallel dimension reduction completed")  # Keep original print statement
    log_and_print("Parallel dimension reduction completed")

    # Compare models in parallel
    print("Starting parallel model comparison")  # Keep original print statement
    select_obj.compare_models_parallel(scaling_methods, dim_reduction_methods, clustering_methods)
    print("Parallel model comparison completed")  # Keep original print statement
    log_and_print("Parallel model comparison completed")

    # Get the results
    print("Getting results")  # Keep original print statement
    results = pd.DataFrame(select_obj.score_dict)

    # Sort results by silhouette score (descending) and Davies-Bouldin score (ascending)
    best_models = results.sort_values(['silhouette_score', 'DB_score'], ascending=[False, True])

    print("Best clustering models:")  # Keep original print statement
    print(best_models.head())  # Keep original print statement
    log_and_print("Best clustering models:")
    log_and_print(best_models.head().to_string())

    # # Optionally, you can plot the clusters for the best model
    # best_model = best_models.iloc[0]
    # select_obj.scaling_data(best_model['norma_method'])
    # select_obj.reduce_dimensions(dimension_method=best_model['dim_red_method'], scaling_method=best_model['norma_method'])
    # labels = select_obj.clustering_data(cluster_method=best_model['clustering_method'])

    # select_obj.plot_clusters(select_obj.reduced_df, labels, title=f"Best Clustering Model: {best_model['clustering_method']}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")  # Keep original print statement
    log_and_print(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # This is needed for Windows support
    main()




