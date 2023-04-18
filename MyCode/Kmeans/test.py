import time
import pandas as pd
from MyCode.Kmeans.KMeans import KMeans

from MyCode.Kmeans.helper import *

label = pd.read_csv('S:/College Folder/UCF/Spring23/ML/HW3/Code/MyCode/Kmeans/dataa/kmeans_data/data.csv').to_numpy()
data = pd.read_csv('S:/College Folder/UCF/Spring23/ML/HW3/Code/MyCode/Kmeans/dataa/kmeans_data/label.csv').to_numpy()

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
        print('=' * 45)
        centroids, clusters = kmeans.fit(arr)

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


dist_sses, dist_accs, dist_times, dist_iters = run(max_iters=1)


print('Q1: Compare the SSEs of Euclidean-K-means, Cosine-K-means, Jarcard-K-means. Which method is better?')
for distance, dist_sse in zip(distances, dist_sses):
  print('{} SSE: {:.3f}'.format(distance.upper(), dist_sse))

print('The best method seems to be', distances[dist_sses.index(min(dist_sses))])

print('\nQ2: Compare the accuracies of Euclidean-K-means Cosine-K-means, Jarcard-K-means. Which method is better?')
for distance, dist_acc in zip(distances, dist_accs):
  print('{} Accuracy: {:.2f}%'.format(distance.upper(), dist_acc*100))

print('The best method seems to be', distances[dist_accs.index(max(dist_accs))])


dist_sses, dist_accs, dist_times, dist_iters = run(max_iters=2, new_stop_criteria=True)


print('Q3:  Which method requires more iterations and times to converge? (New stop criteria)')
for distance, dist_iter, dist_time in zip(distances, dist_iters, dist_times):
  print('{} total iterations: {}, total time taken: {:.2f}s'.format(distance.upper(), dist_iter, dist_time))

print('The best method with least iterations seems to be', distances[dist_iters.index(min(dist_iters))])
print('The best method with least time seems to be', distances[dist_times.index(min(dist_times))])

print('\nQ4: Compare the SSEs of Euclidean-K-means Cosine-K-means, Jarcard-K-means (New stop criteria). Which method is better?')
for distance, dist_sse in zip(distances, dist_sses):
  print('{} SSE: {}'.format(distance.upper(), dist_sse))

print('The best method with least SSE seems to be', distances[dist_sses.index(min(dist_sses))])
