import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import math
from sklearn.cluster import KMeans
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid

_SQRT2 = np.sqrt(2)

def flatten_list(l):
    r = []
    for i in l:
        if type(i) == list or type(i) == np.ndarray:
            for j in i:
                r.append(j)
        else:
            r.append(i)
    return r


def lognormpdf(x, mu, sigma):
    # Compute log N(x; μ, σ²)
    return -0.5 * (((x - mu) / sigma) ** 2 + np.log(2 * np.pi)) - np.log(sigma)


def eq_dist_function(x, y):
    return 1 - int(x == y)


def hellinger_dist(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2


def discretize(data, continuous_attr, bins):
    for attr in continuous_attr:
        column = data[attr].values.reshape(-1, 1)
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
        discrete_col = discretizer.fit_transform(column)
        data[attr+'_disc'] = discrete_col.astype(int)
    return data


def discretize_auto(data, attr, m):
    m = random.sample(range(1, 100), m)
    # calculating k
    k = round(math.log(len(data[attr].unique()),2))
    column = data[attr].values.reshape(-1, 1)
    clusters_m = []
    # run kmeans m times with random seeds
    for m_val in m:
        kmeans = KMeans(n_clusters=k, random_state=m_val).fit(column)
        centers = kmeans.cluster_centers_
        # add all clusters found for this round
        clusters_m.extend(centers)
    # run hierarchical clustering with the clusters found from m rounds
    hier_clustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward').fit(clusters_m)
    classify_points = {}
    # getting the "centroids" from hierarchical clustering labels
    labels = np.asarray(hier_clustering.labels_)
    for (count, label) in enumerate(labels):
        if label in classify_points.keys():
            classify_points[label].append(clusters_m[count][0])
        else:
            classify_points[label] = [clusters_m[count][0]]
    label_items = classify_points.items()
    sorted_items = sorted(label_items)
    hier_centroids = []
    for c in sorted_items:
        lst = c[1]
        hier_centroids.append(lst[len(lst)//2] if lst else None)
    hier_centroids = np.asarray(hier_centroids).reshape(-1, 1)
    kmeans_2 = KMeans(n_clusters=k, init=hier_centroids, n_init=1).fit(column)
    discrete_col = kmeans_2.labels_
    data[attr+'_disc'] = discrete_col.astype(int)
    return data

