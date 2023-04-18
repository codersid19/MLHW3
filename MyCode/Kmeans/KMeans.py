from MyCode.Kmeans.distances import *
from MyCode.Kmeans.helper import *

class KMeans:
    def __init__(self, n_clusters=10, max_iters=10, centroids=None, dist='euclidean', new_stop_criteria=False): #, show_sse=False, show_first_centroid=False, centroid_stop=True):
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
            
            print('Iteration {}/{}: SSE: {} '.format(iter+1, self.max_iters, SSE))

            self.SSEs.append(SSE)
            
            ## stop criteria
            # if the centroids don't change, break
            if self.centroids == old_centroids: break
            if self.new_stop_criteria and i > 0:
                # when the SSE value increases in the next iteration OR when the maximum preset value
                if self.SSEs[iter] > self.SSEs[iter-1]: break

        return self.centroids, clusters
