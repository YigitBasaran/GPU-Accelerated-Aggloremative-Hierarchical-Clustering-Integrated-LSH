import numpy as np
import cupy as cp
import numba as nb
from numba import cuda
import math
import time


@cuda.jit(device=True)
def hash_vector_gpu(vector, random_vectors):
    """Hash a single vector using random projection on the GPU."""
    result = 0
    for i in range(random_vectors.shape[0]):
        dot_product = 0
        for j in range(vector.shape[0]):
            dot_product += vector[j] * random_vectors[i, j]
        if dot_product >= 0:
            result |= 1 << i
    return result

@cuda.jit
def compute_hash_codes_gpu(dataset, random_vectors, hash_codes):
    """Hash the entire dataset using random projection on the GPU."""
    i = cuda.grid(1)  # Global thread index
    if i < dataset.shape[0]:
        hash_codes[i] = hash_vector_gpu(dataset[i], random_vectors)

@cuda.jit
def assign_buckets_gpu(hash_codes, num_buckets, bucket_assignments):
    """Assign hash codes to buckets on the GPU."""
    i = cuda.grid(1)
    if i < hash_codes.size:
        bucket_assignments[i] = hash_codes[i] % num_buckets

@cuda.jit(device=True)
def compute_distance_gpu(vec1, vec2):
    """Compute the Euclidean distance between two vectors on the GPU."""
    distance = 0.0
    for i in range(len(vec1)):
        distance += (vec1[i] - vec2[i]) ** 2
    return math.sqrt(distance)

@cuda.jit
def create_distance_matrix_gpu(flat_data, cluster_offsets, cluster_sizes, distances):
    """Create a distance matrix on the GPU using parallel reduction."""
    shared_mem = cuda.shared.array(shape=0, dtype=nb.float64)  # Dynamically allocated shared memory
    tid = cuda.threadIdx.x
    i, j = cuda.grid(2)
    num_clusters = len(cluster_offsets)

    if i < num_clusters and j < num_clusters:
        if i == j:
            distances[i, j] = cp.inf
        elif i < j:
            sum_distance = 0.0
            count = 0
            cluster1_size = cluster_sizes[i]
            cluster2_size = cluster_sizes[j]

            for p in range(cluster1_size):
                for q in range(cluster2_size):
                    idx1 = cluster_offsets[i] + p
                    idx2 = cluster_offsets[j] + q
                    distance = compute_distance_gpu(flat_data[idx1], flat_data[idx2])
                    
                    # Parallel reduction
                    shared_mem[tid] = distance
                    cuda.syncthreads()

                    step = cuda.blockDim.x // 2
                    while step > 0:
                        if tid < step:
                            shared_mem[tid] += shared_mem[tid + step]
                        step //= 2
                        cuda.syncthreads()

                    if tid == 0:
                        sum_distance += shared_mem[0]
                    count += 1

            if tid == 0:
                avg_distance = sum_distance / count
                distances[i, j] = avg_distance
                distances[j, i] = avg_distance  # Symmetric distance

def flatten_data(data):
    """Flatten the nested list structure into a single list of vectors and return offsets and sizes."""
    flat_data = []
    cluster_offsets = []
    cluster_sizes = []

    offset = 0
    for sublist in data:
        cluster_offsets.append(offset)
        cluster_sizes.append(len(sublist))
        for vec in sublist:
            flat_data.append(vec)
        offset += len(sublist)

    return flat_data, cluster_offsets, cluster_sizes

def create_distance_matrix(data):
    flat_data, cluster_offsets, cluster_sizes = flatten_data(data)
    flat_data = cp.array(flat_data, dtype=cp.float64)

    cluster_offsets = cp.array(cluster_offsets, dtype=cp.int32)
    cluster_sizes = cp.array(cluster_sizes, dtype=cp.int32)
    num_clusters = len(cluster_offsets)

    distances = cp.full((num_clusters, num_clusters), cp.inf, dtype=cp.float64)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(num_clusters / threadsperblock[0])
    blockspergrid_y = math.ceil(num_clusters / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    create_distance_matrix_gpu[blockspergrid, threadsperblock](flat_data, cluster_offsets, cluster_sizes, distances)
    
    start_time = time.time()
    dist = distances.get()
    d2h_copy_time = (time.time()-start_time)*1000

    return dist, d2h_copy_time

def agglomerative_clustering(data, num_clusters, clusters):
    """Perform Agglomerative Hierarchical Clustering on GPU."""
    d2h_copy_time = 0
    if len(clusters) > num_clusters:
        # Initialize distance matrix
        distances, copy_time = create_distance_matrix(data)
        d2h_copy_time += copy_time
    else:
        return clusters, data, d2h_copy_time

    # Merge clusters until the desired number of clusters is reached
    while len(clusters) > num_clusters:
        min_dist = cp.inf
        to_merge = (0, 0)

        # Find the pair of clusters to merge
        for i in range(len(clusters)):
            # Find distance with Single Linkage Method
            idx, dist = min(enumerate(distances[i]), key=lambda x: x[1])

            if dist < min_dist:
                min_dist = dist
                to_merge = (i, idx)

        if to_merge[0] != to_merge[1]:
            # Merge the selected pair of clusters
            new_cluster = clusters[to_merge[0]] + clusters[to_merge[1]]
            clusters.append(new_cluster)

        # Update the dataset with new clusters
        new_data = cp.vstack((data[to_merge[0]], data[to_merge[1]])).tolist()

        data = list(data)
        data.append(new_data)

        # Remove the merged clusters and data
        if to_merge[0] > to_merge[1]:
            del clusters[to_merge[0]]
            del clusters[to_merge[1]]
            del data[to_merge[0]]
            del data[to_merge[1]]
        else:
            del clusters[to_merge[1]]
            del clusters[to_merge[0]]
            del data[to_merge[1]]
            del data[to_merge[0]]


        # Recompute distance matrix
        distances, copy_time = create_distance_matrix(data)
        d2h_copy_time += copy_time

    return clusters, data, d2h_copy_time

def agglomerative_clustering_gpu(data, num_clusters, random_vectors):
    """Wrapper function to measure the elapsed time of agglomerative clustering on GPU."""
    start_time = time.time()
    data = cp.array(data)  # Copy data to the GPU
    random_vectors = cp.array(random_vectors)
    hash_codes = cp.zeros(data.shape[0], dtype=cp.uint64)
    h2d_copy_time = (time.time()-start_time)*1000
    threadsperblock = 256
    blockspergrid = (data.shape[0] + (threadsperblock - 1)) // threadsperblock
    num_buckets = len(data) // num_clusters + num_clusters

    bucketed_data = []
    bucketed_global_data_indices = []

    compute_hash_codes_gpu[blockspergrid, threadsperblock](data, random_vectors, hash_codes)
    start_time = time.time()
    bucket_assignments = cp.zeros(hash_codes.size, dtype=cp.uint32)
    h2d_copy_time += (time.time()-start_time)*1000

    assign_buckets_gpu[blockspergrid, threadsperblock](hash_codes, num_buckets, bucket_assignments)

    bucket_assignments = bucket_assignments.get()
    for bucket in range(num_buckets):
        bucket_indices = np.where(bucket_assignments == bucket)[0].tolist()
        if len(bucket_indices) > 0:
            bucketed_data.append(cp.array(data[bucket_indices]).tolist())
            bucketed_global_data_indices.append(bucket_indices)

    start_time = time.time()
    final_clusters, final_clustered_data, d2h_copy_time = agglomerative_clustering(bucketed_data, num_clusters, bucketed_global_data_indices)
    clustering_time = ((time.time()-start_time)*1000)-d2h_copy_time
    total_copy_time = d2h_copy_time + h2d_copy_time
    return final_clusters, final_clustered_data, clustering_time, total_copy_time
