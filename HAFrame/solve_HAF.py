import numpy as np
from HAFrame.distance import load_distance_matrix


# linear mapping
def map_hdistance_to_cosine_similarity_linear(**kwargs):
    """
        hdistance: K-by-K matrix of pairwise hierarchical distance
    """
    hdistance, min_similarity = kwargs['hdistance'], kwargs['min_similarity']
    assert min_similarity >= -1
    d_max = np.max(hdistance)
    max_similarity = 1.0
    similarity_range = max_similarity-min_similarity
    s = -(similarity_range*hdistance/d_max - max_similarity)
    return s


# exponential decay mapping
def map_hdistance_to_cosine_similarity_exponential_decay(**kwargs):
    hdistance, gamma, min_similarity = kwargs['hdistance'], kwargs['gamma'], kwargs['min_similarity']
    assert min_similarity >= -1
    max_similarity = 1.0
    hd_max = np.max(hdistance)
    similarity_range = max_similarity - min_similarity
    s = similarity_range * np.exp(-gamma*hdistance/hd_max) + min_similarity
    return s


# matrix factorization solver
def cls_weights_matrix_factorization_solver(num_features, num_classes, cos_similarity):

    w, v = np.linalg.eigh(cos_similarity)
    # check for very small eigenvalues
    epsilon = 1e-6
    idx = np.abs(w) < epsilon
    w_prime = w.copy()
    w_prime[idx] = 0
    if np.sum(idx) != 0:
        print(f"very small eigenvalue detected:\t{w[idx]}")

    # check for negative eigenvalues
    if np.min(w_prime) < 0:
        raise ValueError("cosine similarity matrix is not P.S.D.")

    # sqrt of eigenvalues
    d_sqrt = np.sqrt(np.diag(w_prime))

    # random rotation + projection
    tp = np.sqrt(2.0 / num_classes) * np.random.randn(num_features, num_classes)
    q, _ = np.linalg.qr(tp)

    # generate classifier weights
    cls_weights = d_sqrt @ v.T
    cls_weights = q @ cls_weights

    return cls_weights.T


# find max separation with matrix factorization method
def find_max_separation_matrix_factorization_solver(hdistance, num_classes, n_samples, mapping, gamma=None):
    min_sims = np.linspace(-1.0/(num_classes-1), -1.0, n_samples)
    ret_min_sim = float('inf')

    for i, s in enumerate(min_sims):
        if gamma is None:
            cos_sim = mapping(hdistance=hdistance, min_similarity=s)
        else:
            cos_sim = mapping(hdistance=hdistance, gamma=gamma, min_similarity=s)

        # eigendecomposition
        w, v = np.linalg.eigh(cos_sim)

        # check for very small eigenvalues
        epsilon = 1e-6
        idx = np.abs(w) < epsilon
        w_prime = w.copy()
        w_prime[idx] = 0
        if np.sum(idx) != 0:
            print(f"very small eigenvalue detected:\t{w[idx]}")

        # check for negative eigenvalues
        if np.min(w_prime) < 0:
            pass
        else:
            ret_min_sim = min(ret_min_sim, s)

    return ret_min_sim


# find a maximumly separated HAF cls weights for cifar10
def fixed_haf_cls_weights(sargs, num_classes=10, num_features=512):
    hdist = load_distance_matrix(sargs.distance_path, sargs.classes)
    num_samples = 100

    similarity_min = find_max_separation_matrix_factorization_solver(
        hdist,
        num_classes,
        num_samples,
        map_hdistance_to_cosine_similarity_exponential_decay,
        sargs.haf_gamma
    )

    similarity_matrix = map_hdistance_to_cosine_similarity_exponential_decay(
        hdistance=hdist,
        gamma=sargs.haf_gamma,
        min_similarity=similarity_min
    )

    cls_weights = cls_weights_matrix_factorization_solver(num_features, num_classes, similarity_matrix)

    return cls_weights, similarity_matrix, similarity_min


def distance_to_haf_cls_weights(distance_path, classes, num_features, gamma):
    num_classes = len(classes)
    hdist = load_distance_matrix(distance_path, classes)
    num_samples = 100

    similarity_min = find_max_separation_matrix_factorization_solver(
        hdist,
        num_classes,
        num_samples,
        map_hdistance_to_cosine_similarity_exponential_decay,
        gamma
    )

    similarity_matrix = map_hdistance_to_cosine_similarity_exponential_decay(
        hdistance=hdist,
        gamma=gamma,
        min_similarity=similarity_min
    )

    cls_weights = cls_weights_matrix_factorization_solver(num_features, num_classes, similarity_matrix)

    return cls_weights, similarity_matrix, similarity_min


def distance_matrix_to_haf_cls_weights(hdist, classes, num_features, gamma):
    num_classes = hdist.shape[0]
    num_samples = 100

    similarity_min = find_max_separation_matrix_factorization_solver(
        hdist,
        num_classes,
        num_samples,
        map_hdistance_to_cosine_similarity_exponential_decay,
        gamma
    )

    similarity_matrix = map_hdistance_to_cosine_similarity_exponential_decay(
        hdistance=hdist,
        gamma=gamma,
        min_similarity=similarity_min
    )

    cls_weights = cls_weights_matrix_factorization_solver(num_features, num_classes, similarity_matrix)

    return cls_weights, similarity_matrix, similarity_min


def hdistance_to_similarity_matrix(hd_mat, gamma, num_classes=10):
    num_samples = 100

    similarity_min = find_max_separation_matrix_factorization_solver(
        hd_mat,
        num_classes,
        num_samples,
        map_hdistance_to_cosine_similarity_exponential_decay,
        gamma
    )

    similarity_matrix = map_hdistance_to_cosine_similarity_exponential_decay(
        hdistance=hd_mat,
        gamma=gamma,
        min_similarity=similarity_min
    )

    return similarity_matrix
