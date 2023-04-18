import math
import numpy as np

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
