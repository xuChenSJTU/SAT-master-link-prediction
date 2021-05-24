import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import sys
import pickle as pkl
import torch.nn.functional as F
import networkx as nx
import pickle
import os
import scipy.io as sio
from sklearn.svm import SVC
import pandas as pd

def MMD(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def new_load_data(dataset_str, norm_adj=True, generative_flag=False): # {'pubmed', 'citeseer', 'cora'}
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        """Load data."""
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("./data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("./data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = list(range(len(y)))
        idx_val = list(range(len(y), len(y)+500))

        # retutn normalized data
        labels = np.argmax(labels, 1)
        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        if not generative_flag:
            features = normalize_features(features)
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.tocoo().data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

        features = torch.FloatTensor(np.array(features.todense()))
    elif dataset_str in ['steam']:
        freq_item_mat = pickle.load(open(os.path.join(os.getcwd(), 'data', dataset_str, 'freq_item_mat.pkl'), 'rb'))
        features = pickle.load(open(os.path.join(os.getcwd(), 'data', dataset_str, 'sp_fts.pkl'), 'rb'))

        if not generative_flag:
            features = normalize_features(features)

        features = torch.FloatTensor(features.todense())

        adj = freq_item_mat.copy()
        adj[adj < 10.0] = 0.0
        adj[adj >= 10.0] = 1.0
        indices = np.where(adj!=0.0)
        rows = indices[0]
        cols = indices[1]
        values = np.ones(shape=[len(rows)])
        adj = sp.coo_matrix((values, (rows, cols)), shape=[adj.shape[0], adj.shape[1]])
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))

        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

        labels = None
        idx_train = None
        idx_val = None
        idx_test = None
    elif dataset_str == 'facebook_page':
        import json
        edges = pd.read_csv(os.path.join(os.getcwd(), 'data', 'facebook_page', 'musae_facebook_edges.csv'), header=0,
                            sep=',')
        features = json.load(
            open(os.path.join(os.getcwd(), 'data', 'facebook_page', 'musae_facebook_features.json'), 'r'))
        if not generative_flag:
            features = normalize_features(features)
        features = torch.FloatTensor(features)

        # make adj
        adj = sp.coo_matrix((np.ones(len(edges)), (edges.values[:, 0], edges.values[:, 1])),
                            shape=[len(features), len(features)])
        adj = adj.tocsr()
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
        adj = adj.tocoo()
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

        # make labels
        raw_label_data = pd.read_csv(os.path.join(os.getcwd(), 'data', 'facebook_page', 'musae_facebook_target.csv'),
                                     header=0,
                                     sep=',')
        raw_labels = raw_label_data['page_type'].unique()
        label_map = pd.Series(data=range(len(raw_labels)), index=raw_labels)
        raw_label_data['label'] = label_map[raw_label_data['page_type'].values].values
        labels = raw_label_data['label'].values
        labels = torch.LongTensor(labels)
        idx_train = None
        idx_val = None
        idx_test = None
    elif dataset_str == 'coauthor_cs':
        data = load_npz_to_sparse_graph(os.path.join(os.getcwd(), 'data', 'coauthor_cs', 'ms_academic_cs.npz'))

        features = data.attr_matrix.todense()
        if not generative_flag:
            features = normalize_features(features)
        features = torch.FloatTensor(features)

        # make adj
        adj = data.adj_matrix
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)

        adj = adj.tocoo()
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

        # make labels
        labels = torch.LongTensor(data.labels)
        idx_train = None
        idx_val = None
        idx_test = None

    elif dataset_str == 'coauthor_phy':
        data = load_npz_to_sparse_graph(os.path.join(os.getcwd(), 'data', 'coauthor_phy', 'ms_academic_phy.npz'))

        features = data.attr_matrix.todense()

        # binarize the features
        features[np.where(features != 0)] = 1.0
        if not generative_flag:
            features = normalize_features(features)
        features = torch.FloatTensor(features)

        # make adj
        adj = data.adj_matrix
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)

        adj = adj.tocoo()
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

        # make labels
        labels = torch.LongTensor(data.labels)
        idx_train = None
        idx_val = None
        idx_test = None
    elif dataset_str == 'cora_full':
        data = load_npz_to_sparse_graph(os.path.join(os.getcwd(), 'data', 'cora_full', 'cora_full.npz'))

        # delete labeled nodes less than 50
        adj = data.adj_matrix
        features = data.attr_matrix.todense()
        labels = data.labels

        mask = []
        count_dict = {}
        for l in labels:
            tmp_index = np.where(labels == l)[0]
            if l not in count_dict:
                count_dict[l] = len(tmp_index)

            if len(tmp_index) > 55:
                mask.append(True)
            else:
                mask.append(False)
        mask = np.array(mask)

        adj = adj[mask, :][:, mask]
        features = features[mask]
        labels = labels[mask]

        # re-assign labels
        label_map = pd.Series(index=np.unique(labels), data=np.arange(len(np.unique(labels))))
        labels = label_map[labels].values

        # binarize the features
        features[np.where(features != 0)] = 1.0
        if not generative_flag:
            features = normalize_features(features)
        features = torch.FloatTensor(features)

        # make adj
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)

        adj = adj.tocoo()
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

        # make labels
        labels = torch.LongTensor(labels)
        idx_train = None
        idx_val = None
        idx_test = None
    elif dataset_str == 'amazon_computer':
        data = load_npz_to_sparse_graph(os.path.join(os.getcwd(), 'data', 'amazon_computer', 'amazon_electronics_computers.npz'))

        # make adj
        adj = data.adj_matrix
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)

        # make features
        features = data.attr_matrix.todense()

        # binarize the features
        if not generative_flag:
            features = normalize_features(features)
        features = torch.FloatTensor(features)

        # make adj
        adj = data.adj_matrix
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)

        adj = adj.tocoo()
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

        # make labels
        labels = torch.LongTensor(data.labels)
        idx_train = None
        idx_val = None
        idx_test = None
    elif dataset_str == 'amazon_photo':
        data = load_npz_to_sparse_graph(os.path.join(os.getcwd(),'data', 'amazon_photo', 'amazon_electronics_photo.npz'))

        # make adj
        adj = data.adj_matrix
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)

        # make features
        features = data.attr_matrix.todense()

        # binarize the features
        if not generative_flag:
            features = normalize_features(features)
        features = torch.FloatTensor(features)

        # make adj
        adj = data.adj_matrix
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)

        adj = adj.tocoo()
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))

        # make labels
        labels = torch.LongTensor(data.labels)
        idx_train = None
        idx_val = None
        idx_test = None
    elif dataset_str in ['blogcatalog', 'flickr']:
        data = sio.loadmat(os.path.join(os.getcwd(), 'data', dataset_str, '{}.mat'.format(dataset_str)))
        features = data['Attributes'].todense()
        features[features!=0] = 1.0
        features = torch.FloatTensor(features)
        adj = data['Network'].tocoo()
        if norm_adj:
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices = torch.LongTensor(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0))
        values = torch.FloatTensor(adj.data)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))
        labels = data['Label']
        idx_train = None
        idx_val = None
        idx_test = None
    else:
        print('cannot process this dataset !!!')
        raise Exception

    return adj, features, labels, idx_train, idx_val, idx_test



def load_generated_features(path):
    fts = pkl.load(open(path, 'rb'))
    norm_fts = normalize_features(fts)
    norm_fts = torch.FloatTensor(norm_fts)
    return norm_fts

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def cal_accuracy(train_fts, train_lbls, test_fts, test_lbls):
    clf = SVC(gamma='auto')
    clf.fit(train_fts, train_lbls)

    preds_lbls = clf.predict(test_fts)
    acc = accuracy(preds_lbls, test_lbls)
    return acc

def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)

class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.
    """
    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):
        """Create an attributed graph.
        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.
        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)"
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.
        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.
        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.
        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.
        All changes are done inplace.
        """
        G = self.to_unweighted().to_undirected()
        G = eliminate_self_loops(G)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels


import os, sys
import csv
import numpy as np
from numpy import genfromtxt


def subsample(adj, features):
    # input: two sparse matrices

    nb_nodes = adj.shape[0]
    one_hop_features = adj @ features

    features_list = []
    for i in range(nb_nodes):
        features_list.append([])
    idxs0, idxs1 = features.nonzero()
    for i in range(idxs0.shape[0]):
        features_list[idxs0[i]].append(idxs1[i])

    one_hop_features_list = []
    for i in range(nb_nodes):
        one_hop_features_list.append([])
    idxs0, idxs1 = one_hop_features.nonzero()
    for i in range(idxs0.shape[0]):
        one_hop_features_list[idxs0[i]].append(idxs1[i])

    subsample_features_list = []
    for i in range(nb_nodes):
        subsample_features_list.append(list(set(features_list[i]) & set(one_hop_features_list[i])))
    '''

    subsample_features_list = []
    for i in range(nb_nodes):
        subsample_features_list.append(list(set(features_list[i]).difference(set(one_hop_features_list[i]))))
    '''

    for i in range(nb_nodes):
        for ft in subsample_features_list[i]:
            features[i, ft] = 0

    features.eliminate_zeros()

    return features


def output_csv(file_name, score_list):
    with open(file_name, mode='w') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for score in score_list:
            output_writer.writerow([score])


def average_csv(dir, prefix):
    file_list = os.listdir(dir)

    i = 0
    for file in file_list:
        if file.startswith(prefix):
            if i == 0:
                res = np.expand_dims(genfromtxt(dir + '/' + file, delimiter=','), axis=0)
            else:
                res = np.append(res, np.expand_dims(genfromtxt(dir + '/' + file, delimiter=','), axis=0), axis=0)
            i += 1
    print(res.shape)
    print(np.mean(res, axis=0))
    print(np.std(res, axis=0))

