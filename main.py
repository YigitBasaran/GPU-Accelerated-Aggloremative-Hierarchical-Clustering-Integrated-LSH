import numpy as np
from sklearn.datasets import make_blobs
import argparse
import time
from cpu import agglomerative_clustering_cpu
from gpu import agglomerative_clustering_gpu
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import cupy as cp


def main(args):
    X, _ = make_blobs(n_samples=args.num_samples, n_features=args.num_features, centers=5, random_state=42)
    X = X.flatten()  # Convert to 1-dimensional array as specified
    X = X.reshape((args.num_samples, args.num_features))
    random_vectors = np.random.randn(args.num_hash_functions, X.shape[1])




    # Original Aggloremative Clustering silhouette_score calculation
    print("Running Original Aggloremative Clustering...")
    data = X.reshape((args.num_samples, args.num_features))
    agg_cluster = AgglomerativeClustering(n_clusters=args.num_clusters)
    cluster_labels = agg_cluster.fit_predict(data)
    score = silhouette_score(data, cluster_labels)
    print(f'Original Silhouette Score: {score}')


    # CPU implementation
    print("Running on CPU...")
    start_time = time.time()
    cpu_clusters, cpu_clustered_data = agglomerative_clustering_cpu(X, args.num_clusters, random_vectors)
    total_cpu_time = (time.time() - start_time) * 1000
    print("CPU Clusters:", cpu_clusters)
    print("Total CPU Time:", total_cpu_time, " ms")

    # Prepare original data according to CPU clustering
    data_points = []
    labels = []
    for cluster_id, cluster in enumerate(cpu_clustered_data):
        for point in cluster:
            data_points.append(point)
            labels.append(cluster_id)

    data_points = np.array(data_points)
    labels = np.array(labels)

    # Calculate Silhouette score for CPU Implementation
    cpu_score = silhouette_score(data_points, labels)
    print(f'CPU Clustered Data Silhouette Score: {cpu_score}')


    # GPU implementation
    print("Running on GPU...")
    start_time = time.time()
    gpu_clusters, gpu_clustered_data, clustering_gpu_time, copy_gputime = agglomerative_clustering_gpu(X, args.num_clusters, random_vectors)
    total_gpu_time = (time.time() - start_time)*1000
    print("GPU Clusters:", gpu_clusters)
    print("Total GPU Time:", total_gpu_time, " ms")
    print("clustering_gpu_time:", clustering_gpu_time , " ms")
    print("copy_gputime:", copy_gputime , " ms")



    # Prepare original data according to CPU clustering
    data_points = []
    labels = []
    for cluster_id, cluster in enumerate(gpu_clustered_data):
        for point in cluster:
            data_points.append(point)
            labels.append(cluster_id)

    data_points = [cp.asnumpy(point) if isinstance(point, cp.ndarray) else point for point in data_points]
    data_points = np.array(data_points)
    labels = np.array(labels)

    # Calculate Silhouette score for CPU Implementation
    gpu_score = silhouette_score(data_points, labels)
    print(f'GPU Clustered Data Silhouette Score: {gpu_score}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agglomerative Hierarchical Clustering on GPU')
    parser.add_argument('--num_clusters', type=int, default=2, help='Number of clusters')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples in the dataset')
    parser.add_argument('--num_features', type=int, default=10, help='Number of features in the dataset')
    parser.add_argument('--num_hash_functions', type=int, default=5, help='Number of hash functions')
    args = parser.parse_args()
    main(args)