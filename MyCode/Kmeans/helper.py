import matplotlib.pyplot as plt
from collections import defaultdict

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

    arr = sorted(arr, key=lambda x: x[len(arr[0])-1], reverse=False)
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
    labels = {i:0 for i in range(10)}
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
    
        