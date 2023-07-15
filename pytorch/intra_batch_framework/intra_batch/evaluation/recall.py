import torch
import numpy as np
import sklearn.metrics.pairwise

def assign_by_euclidian_at_k(X, T, k, P=None, query=None, gallery=None):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    if X.__class__.__name__ != 'defaultdict':
        print('using sklearn for distance computation')
        distances = sklearn.metrics.pairwise.pairwise_distances(X)
        if k == 1:
            distances = distances + np.eye(distances.shape[0]) * 1000.
    else:
        print('using something for distance computation')
        samples = list(X.keys())
        T = [T[k][k] for k in X.keys()]
        distances = list()
        for k1 in samples:
            dist = list()
            for k2 in samples:
                if k2 in X[k1].keys():
                    dist.append(X[k1][k2])
                else:
                    dist.append(1000)
            distances.append(dist)
        distances = np.array(distances)
    # get nearest points
    print('distances are computed!')
    if k > 1:
        indices = np.argsort(distances, axis = 1)[:, 1 : k + 1]
    else:
        indices = np.expand_dims(np.argmin(distances,  axis=1), axis=1)


    if P is None:
        return np.array([[T[i] for i in ii] for ii in indices]), T
    else:
        return np.array([[T[i] for i in ii] for ii in indices]), T


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    Y = torch.from_numpy(Y)
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))


def assign_by_cos_sim(X, T, k):
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + k)[1][:,1:]]
    Y = Y.float().cpu()
    return Y, T

