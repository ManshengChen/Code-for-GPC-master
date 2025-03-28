from sklearn import preprocessing
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind=np.asarray(ind)
    ind=np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def kmeans(x, y):
    kmeans = KMeans(n_clusters=2, n_init=100)
    y_pred = kmeans.fit_predict(x)
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    return acc, nmi, ari

def spectral_clustering(x, y):
    clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors').fit(x)
    y_pred = clustering.labels_
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    return acc, nmi, ari

def evaluate_embedding(embeddings, labels, search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    acc, nmi, ari = kmeans(x, y)
    print('===== Kmeans performance =====')
    print('Acc: ', acc, 'Nmi: ', nmi, 'Ari: ', ari)
    return acc, nmi, ari

if __name__ == '__main__':
    evaluate_embedding('./data', 'ENZYMES', np.load('tmp/emb.npy'))
