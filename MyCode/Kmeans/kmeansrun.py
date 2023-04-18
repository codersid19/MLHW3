import math
import numpy as np
import time
import pandas as pd
# from KMeans import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict
# from distances import *
# from helper import *

def calculate_centroid(cluster):
    if isinstance(cluster[0][-1], str):
        cluster_len = len(cluster[0]) - 1
    else:
        cluster_len = len(cluster[0])

    centroid = [0] * cluster_len
    for i in range(cluster_len):
        for point in cluster:
            centroid[i] += point[i]
        centroid[i] = centroid[i] / len(cluster)
    return centroid


def plot(clusters, centroid_centers):
    # colors = ["red", "blue", "green"]
    for i, key in enumerate(clusters):
        x, y = [], []
        cluster = clusters[key]
        for c in cluster:
            x.append(c[0])
            y.append(c[1])
        plt.scatter(x, y, marker='o')

    for point in centroid_centers:
        plt.scatter(point[0], point[1], marker='s')

    plt.show()


def draw_and_scatter(clusters, centroid_centers):
    colors = ["red", "blue", "green"]
    for i, key in enumerate(clusters):
        x = []
        y = []
        cluster = clusters[key]
        for c in cluster:
            x.append(c[0])
            y.append(c[1])
        plt.scatter(x, y, marker='^', c=colors[i])

    for point in centroid_centers:
        plt.scatter(point[0], point[1], marker='s')

    plt.show()


def label_cluster(cluster):
    cl = defaultdict(int)
    for point in cluster:
        cl[point[-1]] += 1
    return cl


def get_target_labels(data, label):
    arr = []

    for i, row_item in enumerate(data):
        temp = []
        for j, col_item in enumerate(row_item):
            temp.append(data[i][j])
        temp.append(label[i][0])
        arr.append(temp)

    arr = sorted(arr, key=lambda x: x[len(arr[0]) - 1], reverse=False)
    return dict(label_cluster(arr))


def get_accuracy(labels, target_labels):
    total = 0
    mismatch = 0

    for target_label in target_labels:
        total += target_labels[target_label]
        mismatch += abs(target_labels[target_label] - labels[target_label])

    accuracy = (total - mismatch) / total
    return accuracy


def get_labels(clusters):
    labels = {i: 0 for i in range(10)}
    for key in clusters:
        d = dict(label_cluster(clusters[key]))
        mx, s = 0, 0
        label = ''
        for k in d:
            s += d[k]
            if d[k] > mx:
                mx = d[k]
                label = k
            labels[label] = mx

    return labels



def euclidean_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1-p2)

def cosine_similarity(p1, p2):
    A = np.array(p1)
    B = np.array(p2)
    return 1 - np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def jaccard(p1, p2):
    min_sum = np.sum(np.minimum(p1, p2), axis = 0)
    max_sum = np.sum(np.maximum(p1, p2), axis = 0)
    return 1 - (min_sum/max_sum)


class KMeans:
    def __init__(self, n_clusters=10, max_iters=10, centroids=None, dist='euclidean',
                 new_stop_criteria=False):  # , show_sse=False, show_first_centroid=False, centroid_stop=True):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = centroids
        self.new_stop_criteria = new_stop_criteria
        self.SSEs = []
        if dist == 'euclidean':
            self.distance = euclidean_distance
        elif dist == 'cosine':
            self.distance = cosine_similarity
        elif dist == 'jaccard':
            self.distance = jaccard

    def init_centroids(self):
        random_choice = np.random.choice(range(len(self.data)), self.n_clusters, replace=False)
        centroids = []
        for choice in random_choice:
            if isinstance(self.data[choice][-1], str):
                centroids.append(self.data[choice][:-1])
            else:
                centroids.append(self.data[choice])
        return centroids

    def fit(self, data):
        self.data = data
        if self.centroids is None:
            self.centroids = self.init_centroids()

        for iter in range(self.max_iters):
            clusters = defaultdict(list)
            SSE = 0

            # classifying each point in the data to the nearest cluster
            for point in data:
                # init the temporary centroid and the minimum distance
                current_centroid = -1
                min_dist = 99999
                # calculate the distance of the current point with all the centroids
                # assign the point to the centroid with the lowest distance
                for i, centroid in enumerate(self.centroids):
                    dist = self.distance(point, centroid)
                    if dist < min_dist:
                        current_centroid = i
                        min_dist = dist

                clusters[current_centroid].append(point)

            old_centroids = self.centroids.copy()
            # recalculation of centroids
            for key in clusters.keys():
                self.centroids[key] = calculate_centroid(clusters[key])

            for key in clusters.keys():
                cluster = clusters[key]
                centroid_point = self.centroids[key]

                for cluster_point in cluster:
                    SSE += euclidean_distance(centroid_point, cluster_point)

            print('Iteration {}/{}: SSE: {} '.format(iter + 1, self.max_iters, SSE))

            self.SSEs.append(SSE)

            ## stop criteria
            # if the centroids don't change, break
            if self.centroids == old_centroids: break
            if self.new_stop_criteria and i > 0:
                # when the SSE value increases in the next iteration OR when the maximum preset value
                if self.SSEs[iter] > self.SSEs[iter - 1]: break

        return self.centroids, clusters



label = pd.read_csv('S:/College Folder/UCF/Spring23/ML/HW3/Code/MyCode/Kmeans/dataa/kmeans_data/label.csv').to_numpy()
data = pd.read_csv('S:/College Folder/UCF/Spring23/ML/HW3/Code/MyCode/Kmeans/dataa/kmeans_data/data.csv').to_numpy()

arr = []
for row in range(len(data)):
  temp = []
  for col in range(len(data[row])):
    temp.append(data[row][col])
  temp.append(label[row][0])
  arr.append(temp)

arr = sorted(arr, key=lambda x: x[len(arr[0])-1], reverse=False)


target_labels = dict(label_cluster(arr))
print(target_labels)

distances = ['euclidean', 'cosine', 'jaccard']


def run(max_iters=10, new_stop_criteria=False):
    dist_sses = []
    dist_accs = []
    dist_times = []
    dist_iters = []
    for dist in distances:
        start_time = time.time()
        print(('=' * 15) + ' ' + dist + ' ' + ('=' * 15))
        kmeans = KMeans(dist=dist, max_iters=max_iters, new_stop_criteria=new_stop_criteria)

        centroids, clusters = kmeans.fit(arr)
        print('=' * 45)

        labels = get_labels(clusters)

        # plot(clusters, centroids)

        time_taken = time.time() - start_time
        print('\n{} STATS:'.format(dist.upper()))
        print('Total time taken: {}'.format(time_taken))
        print('SSE = ', kmeans.SSEs[-1])
        print('Accuracy = {:3f}'.format(get_accuracy(labels, target_labels)))

        print('Original Labels: ', target_labels)
        print('Predicted Labels: ', labels)
        dist_sses.append(kmeans.SSEs[-1])
        dist_accs.append(get_accuracy(labels, target_labels))
        dist_times.append(time_taken)
        dist_iters.append(len(kmeans.SSEs))

    return dist_sses, dist_accs, dist_times, dist_iters

dist_sses, dist_accs, dist_times, dist_iters = run(max_iters=50)


print('Q1: Compare the SSEs of Euclidean-K-means, Cosine-K-means, Jarcard-K-means. Which method is better?')
for distance, dist_sse in zip(distances, dist_sses):
  print('{} SSE: {:.3f}'.format(distance.upper(), dist_sse))

print('The best method seems to be', distances[dist_sses.index(min(dist_sses))])

print('\nQ2: Compare the accuracies of Euclidean-K-means Cosine-K-means, Jarcard-K-means. Which method is better?')
for distance, dist_acc in zip(distances, dist_accs):
  print('{} Accuracy: {:.2f}%'.format(distance.upper(), dist_acc*100))

print('The best method seems to be', distances[dist_accs.index(max(dist_accs))])


dist_sses, dist_accs, dist_times, dist_iters = run(max_iters=50, new_stop_criteria=True)

print('Q3:  Which method requires more iterations and times to converge? (New stop criteria)')
for distance, dist_iter, dist_time in zip(distances, dist_iters, dist_times):
    print('{} total iterations: {}, total time taken: {:.2f}s'.format(distance.upper(), dist_iter, dist_time))

print('The best method with least iterations seems to be', distances[dist_iters.index(min(dist_iters))])
print('The best method with least time seems to be', distances[dist_times.index(min(dist_times))])

print(
    '\nQ4: Compare the SSEs of Euclidean-K-means Cosine-K-means, Jarcard-K-means (New stop criteria). Which method is better?')
for distance, dist_sse in zip(distances, dist_sses):
    print('{} SSE: {}'.format(distance.upper(), dist_sse))

print('The best method with least SSE seems to be', distances[dist_sses.index(min(dist_sses))])


# [len(x) for x in dist_sses]
dist_sses