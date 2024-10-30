import numpy as np
from sklearn.datasets import make_blobs
import argparse
import time
from cpu import agglomerative_clustering_cpu
from gpu import agglomerative_clustering_gpu
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt

# Function to run the main process
def run_clustering(num_clusters, num_samples, num_features, num_hash_functions):
    X, _ = make_blobs(n_samples=num_samples, n_features=num_features, centers=5, random_state=42)
    X = X.flatten()  # Convert to 1-dimensional array as specified
    X = X.reshape((num_samples, num_features))
    random_vectors = np.random.randn(num_hash_functions, X.shape[1])

    # Original Aggloremative Clustering silhouette_score calculation
    data = X.reshape((num_samples, num_features))
    agg_cluster = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = agg_cluster.fit_predict(data)
    original_score = silhouette_score(data, cluster_labels)

    # CPU implementation
    start_time = time.time()
    cpu_clusters, cpu_clustered_data = agglomerative_clustering_cpu(X, num_clusters, random_vectors)
    total_cpu_time = (time.time() - start_time) * 1000

    # Prepare data according to CPU clustering for silhouette score
    data_points = []
    labels = []
    for cluster_id, cluster in enumerate(cpu_clustered_data):
        for point in cluster:
            data_points.append(point)
            labels.append(cluster_id)

    data_points = np.array(data_points)
    labels = np.array(labels)
    cpu_score = silhouette_score(data_points, labels)

    # GPU implementation
    start_time = time.time()
    gpu_clusters, gpu_clustered_data, clustering_gpu_time, copy_gputime = agglomerative_clustering_gpu(X, num_clusters, random_vectors)
    total_gpu_time = (time.time() - start_time) * 1000

    # Prepare data according to GPU clustering for silhouette score
    data_points = []
    labels = []
    for cluster_id, cluster in enumerate(gpu_clustered_data):
        for point in cluster:
            data_points.append(point)
            labels.append(cluster_id)

    data_points = [cp.asnumpy(point) if isinstance(point, cp.ndarray) else point for point in data_points]
    data_points = np.array(data_points)
    labels = np.array(labels)
    gpu_score = silhouette_score(data_points, labels)

    # Calculate average silhouette score of CPU and GPU
    average_score = (cpu_score + gpu_score) / 2

    return {
        'num_samples': num_samples,
        'num_features': num_features,
        'num_clusters': num_clusters,
        'num_hash_functions': num_hash_functions,
        'cpu_total_time': total_cpu_time,
        'gpu_copy_time': copy_gputime,
        'gpu_total_time': total_gpu_time,
        'original_silhouette_score': original_score,
        'average_silhouette_score': average_score,
        'clustering_gpu_time': clustering_gpu_time,
        'copy_gputime': copy_gputime
    }

# Parameters to iterate over
num_samples_list = [100, 500]
num_clusters_list = [2, 50]
num_features_list = [5, 20]
num_hash_functions_list = [4, 12]

# Collect results
results = []
i = 0
for num_samples in num_samples_list:
    for num_features in num_features_list:
        for num_clusters in num_clusters_list:
            for num_hash_functions in num_hash_functions_list:
                i += 1
                print(f"Sample: {num_samples} features: {num_features} clusters: {num_clusters} hash fncs: {num_hash_functions} ")
                result = run_clustering(num_clusters, num_samples, num_features, num_hash_functions)
                results.append(result)

# Create DataFrame for results
df = pd.DataFrame(results)

# Sort DataFrame
df.sort_values(by=['num_samples', 'num_features', 'num_clusters', 'num_hash_functions'], ascending=[True, True, False, True], inplace=True)

# Save DataFrame to CSV
df.to_csv('clustering_results_new.csv', index=False)

# Display DataFrame
print(df)

# Compute average GPU times
average_copy_time = df['gpu_copy_time'].mean()
average_clustering_time = df['clustering_gpu_time'].mean()
average_total_time = df['gpu_total_time'].mean()

# Determine the 'Other Processes' time
other_processes_time = average_total_time - (average_copy_time + average_clustering_time)
if other_processes_time < 0:
    other_processes_time = 0  # Ensure no negative time

# Pie chart for average GPU times
labels = 'Clustering Time', 'Copy Time', 'Other Processes'
sizes = [average_clustering_time, average_copy_time, other_processes_time]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Average GPU Times')
plt.show()
fig1.savefig("avgGpuTimesPieChart.png")


# Plot for varying parameters
for param_name in ['num_samples', 'num_features', 'num_clusters', 'num_hash_functions']:
    if param_name == 'num_samples':
        other_params = df[(df['num_features'] == 20) & (df['num_clusters'] == 2) & (df['num_hash_functions'] == 12)]
    elif param_name == 'num_features':
        other_params = df[(df['num_samples'] == 500) & (df['num_clusters'] == 2) & (df['num_hash_functions'] == 12)]
    elif param_name == 'num_clusters':
        other_params = df[(df['num_samples'] == 500) & (df['num_features'] == 20) & (df['num_hash_functions'] == 12)]
    elif param_name == 'num_hash_functions':
        other_params = df[(df['num_samples'] == 500) & (df['num_features'] == 20) & (df['num_clusters'] == 2)]

    plt.figure(figsize=(10, 6))
    plt.plot(other_params[param_name], other_params['cpu_total_time'], label='CPU Total Time', marker='o')
    plt.plot(other_params[param_name], other_params['gpu_total_time'], label='GPU Total Time', marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Time (ms)')
    plt.title(f'CPU and GPU Times vs {param_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./{param_name}.png")
    plt.show()
