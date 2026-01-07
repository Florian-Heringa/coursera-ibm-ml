# Unsupervised Leaning Notes



## Clustering

### Common Algorithms
- K-means (inputs => K)
    - Choose K, which determines how many centroids to try to place in the data.
    - Place the K centroids randomly
    - Iteratively determine which centroid each point belongs to by distance
    - Then update the centroid position to the mean of these points
    - Repeat until the assignments of points to clusters does not change anymore (equivalent to the centroids staying at the same position)
- Mixture Of Gaussians (inputs => n_components, covariance_type)
    - Assumes that the clusters follow a gaussian distribution with defined mean and covariance
    - Fits K gaussians to the dataset.
- Hierarchical Agglomerative Clustering (inputs => K, distance_metric, linkage_criteria)
    - Start with a pair of points that have the closest distance
    - Repeat for all points
    - Combine clusters with a specified likage criteria
    - Repeat until minimum cluster size is above a threshold
- DBSCAN (inputs => distance_metric, epsilon (radius), min_samples)
    - Density Based Spatial Clustering of Application with Noise
    - Can handle outliers as 'noise' points
    - Find core points by threshold, more than min_samples within a window distance_metric(epsilon).
    - Update each point to be a core point, reachable point or noise point.
    - If no more points are reachable from an iteration step, pick a random non-visited point and start a new cluster.
- Mean Shift (inputs => bandwidth)
    - Hill climbing algorithm (finds the location in the cluster with the highest density.)
    - Calculates the weighted mean in a local neighborhood for each point. Then update the location of the centroid to the mean position. 
    - Keep iterating until the location converges.
    - Repeat for each point, then group points with the same convergence point into the same cluster.

A good overview of the clustering methods available in scikit-learn can be found [here](https://scikit-learn.org/stable/modules/clustering.html).

### Distance Metrics
- Manhattan (abs)
- Euclidean (squared)
- Cosine (direction) 
- Jaccard (set)

## Dimensionality Reduction

### Common Algorithms
- PCA
    - A method to reduce the dimension of a data space by finding principal values
    - The principal values correspond to the Singular Value Decomposition result which yields these as orthogonal vectors, together with how much of the original variance of the data they explain.
    - Usually a variance cutoff is selected (like 95%) up to which components are used. Components that supply minimal variance can be discarded.