import numpy as np
import numba as nb
import time

@nb.njit(nopython=True)
def hash_vector(vector, random_vectors):
    """Hash a single vector using random projection."""
    result = 0
    for i, random_vector in enumerate(random_vectors):
        if np.dot(vector, random_vector) >= 0:
            result |= 1 << i
    return result

@nb.njit(nopython=True)
def hash_dataset(dataset, random_vectors):
    """Hash the entire dataset using random projection."""
    num_vectors = len(dataset)
    hash_codes = np.zeros((num_vectors,), dtype=np.uint64)

    for i in nb.prange(num_vectors):
        hash_codes[i] = hash_vector(dataset[i], random_vectors)

    return hash_codes

@nb.njit(nopython=True)
def compute_hash_codes(dataset, random_vectors):
    """Compute hash codes for the dataset using random projection."""
    hash_codes = hash_dataset(dataset, random_vectors)

    return hash_codes

@nb.njit(nopython=True)
def assign_buckets(hash_codes, num_buckets):
    """Assign hash codes to buckets."""
    num_vectors = len(hash_codes)
    bucket_assignments = np.zeros((num_vectors,), dtype=np.uint32)

    for i in nb.prange(num_vectors):
        bucket_assignments[i] = hash_codes[i] % num_buckets

    return bucket_assignments


def create_distance_matrix(data):
    num_points = len(data)
    distances = np.full((num_points, num_points), np.inf)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distances[i, j] = compute_distance(data[i], data[j])
            distances[j, i] = distances[i, j]  # Symmetric distance
    return distances

def compute_distance(vec1, vec2):
    """Compute the euclidian distance between two vectors (or sets of vectors)."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.min([np.linalg.norm(p1 - p2) for p1 in vec1 for p2 in vec2])

def compute_centroid_of_cluster(cluster):
    """Compute the centroid of a cluster"""
    # Initialize a list with zeros of the same length as the cluster elements
    centroid = [0] * len(cluster[0])

    # Sum each element of the cluster element-wise
    for element in cluster:
        for i in range(len(element)):
            centroid[i] += element[i]

    # Divide by the number of elements in the cluster to get the mean
    num_elements = len(cluster)
    for i in range(len(centroid)):
        centroid[i] /= num_elements

    return centroid

def agglomerative_clustering(data, num_clusters,clusters):
    """Perform Agglomerative Hierarchical Clustering on CPU."""

    if(len(clusters) > num_clusters):
        # Initialize distance matrix
        distances = create_distance_matrix(data)
    else:
        return clusters, data

    # Merge clusters until the desired number of clusters is reached
    while len(clusters) > num_clusters:
        min_dist = np.inf
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
        new_data = np.vstack((data[to_merge[0]], data[to_merge[1]])).tolist()

        if(type(data) != type(list())):
            data = data.tolist()
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
        distances = create_distance_matrix(data)

    return (clusters, data)


def agglomerative_clustering_cpu(data, num_clusters, random_vectors):
    """Wrapper function to measure the elapsed time of agglomerative clustering on CPU."""

    bucketed_data = []
    bucketed_global_data_indices= []
    num_buckets = len(data) // num_clusters + num_clusters


    hash_codes = compute_hash_codes(data, random_vectors)
    bucket_assignments = assign_buckets(hash_codes, num_buckets)
    # print(hash_codes)

    for bucket in range(num_buckets):
        bucket_indices = np.where(bucket_assignments == bucket)[0].tolist()
        if len(bucket_indices) > 0:
            bucketed_data.append(data[bucket_indices].tolist())
            bucketed_global_data_indices.append(bucket_indices)

    final_clusters, final_clustered_data = agglomerative_clustering(bucketed_data, num_clusters, bucketed_global_data_indices)



    return final_clusters, final_clustered_data