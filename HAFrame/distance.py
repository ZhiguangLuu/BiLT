import lzma
import pickle
import numpy as np

__all__ = [
    'distance_dict_to_mat',
    'load_distance_matrix',
    'load_distances'
]


class DistanceDict(dict):
    """
    Small helper class implementing a symmetrical dictionary to hold distance data.
    """

    def __init__(self, distances):
        self.distances = {tuple(sorted(t)): v for t, v in distances.items()}

    def __getitem__(self, i):
        if i[0] == i[1]:
            return 0
        else:
            return self.distances[(i[0], i[1]) if i[0] < i[1] else (i[1], i[0])]

    def __setitem__(self, i):
        raise NotImplementedError()


def load_distances(data_path):
    """
    Load the distances corresponding to a given hierarchy.
    Args:
        data_path: path of the distance dictionary {('noda_a','node_b'):dist}
    Returns:
        A dictionary of the form {(wnid1, wnid2): distance} for all precomputed
        distances for the hierarchy.
    """
    with lzma.open(data_path, "rb") as f:
        return DistanceDict(pickle.load(f))


def load_distance_matrix(data_path, class_names):
    hdict = load_distances(data_path)
    hd_mat = np.array([[hdict[c1, c2] for c1 in class_names] for c2 in class_names])
    return hd_mat


def distance_dict_to_mat(distance_dict, class_names):
    num_classes = len(class_names)
    distance_matrix = np.zeros([num_classes, num_classes])
    for r in range(num_classes):
        for c in range(num_classes):
            # print(f"distance[({classes[r]},{classes[c]})]:{distance[(classes[r], classes[c])]}")
            distance_matrix[r, c] = distance_dict[(class_names[r], class_names[c])]
    return distance_matrix
    
