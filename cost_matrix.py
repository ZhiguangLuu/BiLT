import nltk
from nltk.tree import Tree
from collections import deque, defaultdict
from MBM.better_mistakes.trees import load_hierarchy
import numpy as np
import torch
import pickle
import os

def loss_ratio_scheduler(sargs, epoch):
    """
    :param sargs: parsed input arguments
    :param epoch: current epoch number
    :return:
        gamma: mixing weights for optimal transport loss
    """
    if epoch > sargs.start_opl:
        if sargs.loss_schedule == 'linear-increase':
            gamma = (epoch-sargs.start_opl) / (sargs.epochs-sargs.start_opl)
        elif sargs.loss_schedule == 'cosine':
            p = float(sargs.loss_schedule_period)
            gamma = 0.5*(1 - np.cos(2 * (np.pi / p) * epoch))
        elif sargs.loss_schedule == 'cosine-linear-increase':
            p = float(sargs.loss_schedule_period)
            gamma = 0.5 * (1 - np.cos(2 * (np.pi / p) * epoch)) * epoch / sargs.epochs
        elif sargs.loss_schedule == 'step':
            p = int(float(sargs.loss_schedule_period) * sargs.epochs)
            gamma = 1.0 * ((epoch-sargs.start_opl + p) // p)
        elif sargs.loss_schedule == 'constant':
            gamma = 1.0
        else:
            raise ValueError(f"Unknown loss schedule: {sargs.loss_schedule}")
    else:
        gamma = 0.0
    return gamma

def load_graph_imagenet(save_path):
    with open(save_path, "rb") as file:
        trees = pickle.load(file)
    graph = defaultdict(list)
    if isinstance(trees, np.ndarray):
        trees = trees.tolist()
    level = 12
    for i in range(len(trees)):
        trees[i][-1] = 'root'
    for i in range(len(trees)):
        for j in range(level):
            if j == level-1:
                if ('L' + str(level-j) + '-' + str(trees[i][j])) not in graph['root']:
                    graph['root'].append('L' + str(level-j) + '-' + str(trees[i][j]))
                if 'root' not in graph['L' + str(level-j) + '-' + str(trees[i][j])]:
                    graph['L' + str(level-j) + '-' + str(trees[i][j])].append('root')
            else:
                if ('L' + str(level-j-1) + '-' + str(trees[i][j+1])) not in graph['L' + str(level-j) + '-' + str(trees[i][j])]:
                    graph['L' + str(level-j) + '-' + str(trees[i][j])].append('L' + str(level-j-1) + '-' + str(trees[i][j+1]))
                if ('L' + str(level-j) + '-' + str(trees[i][j])) not in graph['L' + str(level-j-1) + '-' + str(trees[i][j+1])]:
                    graph['L' + str(level-j-1) + '-' + str(trees[i][j+1])].append('L' + str(level-j) + '-' + str(trees[i][j]))
    return graph

def load_graph(save_path):
    with open(save_path, "rb") as file:
        trees = pickle.load(file)
    graph = defaultdict(list)
    if isinstance(trees, np.ndarray):
        trees = trees.tolist()
    level = len(trees[0])
    graph['root'] = []
    for i in range(len(trees)):
        trees[i].insert(level, 'root')
        for j in range(level):
            if j == level-1:
                if ('L' + str(level-j) + '-' + str(trees[i][j])) not in graph['root']:
                    graph['root'].append('L' + str(level-j) + '-' + str(trees[i][j]))
                if 'root' not in graph['L' + str(level-j) + '-' + str(trees[i][j])]:
                    graph['L' + str(level-j) + '-' + str(trees[i][j])].append('root')
            else:
                if ('L' + str(level-j-1) + '-' + str(trees[i][j+1])) not in graph['L' + str(level-j) + '-' + str(trees[i][j])]:
                    graph['L' + str(level-j) + '-' + str(trees[i][j])].append('L' + str(level-j-1) + '-' + str(trees[i][j+1]))
                if ('L' + str(level-j) + '-' + str(trees[i][j])) not in graph['L' + str(level-j-1) + '-' + str(trees[i][j+1])]:
                    graph['L' + str(level-j-1) + '-' + str(trees[i][j+1])].append('L' + str(level-j) + '-' + str(trees[i][j]))
    return graph

def hierarchy_to_graph(tree):
    graph = defaultdict(list)
    for subtree in tree.subtrees():
        for child in subtree:
            if isinstance(child, Tree):
                graph[subtree.label()].append(child.label())
                graph[child.label()].append(subtree.label())
            else:
                graph[subtree.label()].append(child)
                graph[child].append(subtree.label())
    return graph

def bfs_shortest_path(graph, start, goal):
    
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    raise ValueError(f'No path from {start} to {goal}')

def cal_cost_matrix(graph, finecls, coarsecls, finelevel):
    Cost_matrix = np.zeros((finecls, coarsecls))
    for i in range(finecls):
        start = f'L{finelevel}-{i}'
        for j in range(coarsecls):
            if finecls == coarsecls:
                end = f'L{finelevel}-{j}'
            else:
                end = f'L{finelevel-1}-{j}'
            
            if start == end:
                Cost_matrix[i][j] = 0
            else:
                dis = bfs_shortest_path(graph, start, end)
                Cost_matrix[i][j] = len(dis)//2
    return Cost_matrix

def cifar100_cost_matrix():
    hierarchy = load_hierarchy('cifar-100', 'data/cifar-l5/original')
    # 100, 20, 8, 4, 2
    classes = [100, 20, 8, 4, 2]
    graph = hierarchy_to_graph(hierarchy)
    cifar100_Cost_matrix_l5 = cal_cost_matrix(graph, classes[0], classes[0], 5)
    cifar100_Cost_matrix_l4_0 = cal_cost_matrix(graph, classes[0], classes[1], 5)
    cifar100_Cost_matrix_l4_1 = cal_cost_matrix(graph, classes[1], classes[1], 4)
    cifar100_Cost_matrix_l3_0 = cal_cost_matrix(graph, classes[1], classes[2], 4)
    cifar100_Cost_matrix_l3_1 = cal_cost_matrix(graph, classes[2], classes[2], 3)
    cifar100_Cost_matrix_l2_0 = cal_cost_matrix(graph, classes[2], classes[3], 3)
    cifar100_Cost_matrix_l2_1 = cal_cost_matrix(graph, classes[3], classes[3], 2)
    cifar100_Cost_matrix_l1_0 = cal_cost_matrix(graph, classes[3], classes[4], 2)
    cifar100_Cost_matrix_l1_1 = cal_cost_matrix(graph, classes[4], classes[4], 1)

    return [cifar100_Cost_matrix_l5, cifar100_Cost_matrix_l4_0, cifar100_Cost_matrix_l4_1, cifar100_Cost_matrix_l3_0, cifar100_Cost_matrix_l3_1, cifar100_Cost_matrix_l2_0, cifar100_Cost_matrix_l2_1, cifar100_Cost_matrix_l1_0, cifar100_Cost_matrix_l1_1]

def aircraft_cost_matrix():
    save_path = "_fgvc_aircraft/fgvc_aircraft_tree_list_level3.pkl"
    graph = load_graph(save_path)
    # 100, 70, 30
    classes = [100, 70, 30]
    aircraft_Cost_matrix_l3 = cal_cost_matrix(graph, classes[0], classes[0], 3)
    aircraft_Cost_matrix_l2_0 = cal_cost_matrix(graph, classes[0], classes[1], 3)
    aircraft_Cost_matrix_l2_1 = cal_cost_matrix(graph, classes[1], classes[1], 2)
    aircraft_Cost_matrix_l1_0 = cal_cost_matrix(graph, classes[1], classes[2], 2)
    aircraft_Cost_matrix_l1_1 = cal_cost_matrix(graph, classes[2], classes[2], 1)
    return [aircraft_Cost_matrix_l3, aircraft_Cost_matrix_l2_0, aircraft_Cost_matrix_l2_1, aircraft_Cost_matrix_l1_0, aircraft_Cost_matrix_l1_1]

def inat19_cost_matrix():
    save_path = "_iNat19/inat19_tree_list_level7.pkl"
    graph = load_graph(save_path)
    # 1010, 72, 57, 34, 9, 4, 3
    classes = [1010, 72, 57, 34, 9, 4, 3]
    inat19_Cost_matrix_l7 = cal_cost_matrix(graph, classes[0], classes[0], 7)
    inat19_Cost_matrix_l6_0 = cal_cost_matrix(graph, classes[0], classes[1], 7)
    inat19_Cost_matrix_l6_1 = cal_cost_matrix(graph, classes[1], classes[1], 6)
    inat19_Cost_matrix_l5_0 = cal_cost_matrix(graph, classes[1], classes[2], 6)
    inat19_Cost_matrix_l5_1 = cal_cost_matrix(graph, classes[2], classes[2], 5)
    inat19_Cost_matrix_l4_0 = cal_cost_matrix(graph, classes[2], classes[3], 5)
    inat19_Cost_matrix_l4_1 = cal_cost_matrix(graph, classes[3], classes[3], 4)
    inat19_Cost_matrix_l3_0 = cal_cost_matrix(graph, classes[3], classes[4], 4)
    inat19_Cost_matrix_l3_1 = cal_cost_matrix(graph, classes[4], classes[4], 3)
    inat19_Cost_matrix_l2_0 = cal_cost_matrix(graph, classes[4], classes[5], 3)
    inat19_Cost_matrix_l2_1 = cal_cost_matrix(graph, classes[5], classes[5], 2)
    inat19_Cost_matrix_l1_0 = cal_cost_matrix(graph, classes[5], classes[6], 2)
    inat19_Cost_matrix_l1_1 = cal_cost_matrix(graph, classes[6], classes[6], 1)
    iNat19_cost_matrices = [inat19_Cost_matrix_l7, inat19_Cost_matrix_l6_0, inat19_Cost_matrix_l6_1, inat19_Cost_matrix_l5_0, inat19_Cost_matrix_l5_1, inat19_Cost_matrix_l4_0, inat19_Cost_matrix_l4_1, inat19_Cost_matrix_l3_0, inat19_Cost_matrix_l3_1, inat19_Cost_matrix_l2_0, inat19_Cost_matrix_l2_1, inat19_Cost_matrix_l1_0, inat19_Cost_matrix_l1_1]
    if not os.path.exists('_iNat19/iNat19_cost_matrices.pkl'):
        with open('_iNat19/iNat19_cost_matrices.pkl', 'wb') as file:
            pickle.dump(iNat19_cost_matrices, file)
    return iNat19_cost_matrices


def imagenet_cost_matrix():
    save_path = "_tiered_imagenet/tiered_tree_list_level13.pkl"
    graph = load_graph_imagenet(save_path)
    # 608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2
    classes = [608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2]
    imagenet_Cost_matrix_l12 = cal_cost_matrix(graph, classes[0], classes[0], 12)
    imagenet_Cost_matrix_l11_0 = cal_cost_matrix(graph, classes[0], classes[1], 12)
    imagenet_Cost_matrix_l11_1 = cal_cost_matrix(graph, classes[1], classes[1], 11)
    imagenet_Cost_matrix_l10_0 = cal_cost_matrix(graph, classes[1], classes[2], 11)
    imagenet_Cost_matrix_l10_1 = cal_cost_matrix(graph, classes[2], classes[2], 10)
    imagenet_Cost_matrix_l9_0  = cal_cost_matrix(graph, classes[2], classes[3], 10)
    imagenet_Cost_matrix_l9_1  = cal_cost_matrix(graph, classes[3], classes[3], 9)
    imagenet_Cost_matrix_l8_0  = cal_cost_matrix(graph, classes[3], classes[4], 9)
    imagenet_Cost_matrix_l8_1  = cal_cost_matrix(graph, classes[4], classes[4], 8)
    imagenet_Cost_matrix_l7_0  = cal_cost_matrix(graph, classes[4], classes[5], 8)
    imagenet_Cost_matrix_l7_1  = cal_cost_matrix(graph, classes[5], classes[5], 7)
    imagenet_Cost_matrix_l6_0  = cal_cost_matrix(graph, classes[5], classes[6], 7)
    imagenet_Cost_matrix_l6_1  = cal_cost_matrix(graph, classes[6], classes[6], 6)
    imagenet_Cost_matrix_l5_0  = cal_cost_matrix(graph, classes[6], classes[7], 6)
    imagenet_Cost_matrix_l5_1  = cal_cost_matrix(graph, classes[7], classes[7], 5)
    imagenet_Cost_matrix_l4_0  = cal_cost_matrix(graph, classes[7], classes[8], 5)
    imagenet_Cost_matrix_l4_1  = cal_cost_matrix(graph, classes[8], classes[8], 4)
    imagenet_Cost_matrix_l3_0  = cal_cost_matrix(graph, classes[8], classes[9], 4)
    imagenet_Cost_matrix_l3_1  = cal_cost_matrix(graph, classes[9], classes[9], 3)
    imagenet_Cost_matrix_l2_0  = cal_cost_matrix(graph, classes[9], classes[10],3)
    imagenet_Cost_matrix_l2_1  = cal_cost_matrix(graph, classes[10],classes[10],2)
    imagenet_Cost_matrix_l1_0  = cal_cost_matrix(graph, classes[10],classes[11],2)
    imagenet_Cost_matrix_l1_1  = cal_cost_matrix(graph, classes[11],classes[11],1)
    imagenet_cost_matrices = [imagenet_Cost_matrix_l12, imagenet_Cost_matrix_l11_0, imagenet_Cost_matrix_l11_1, imagenet_Cost_matrix_l10_0, imagenet_Cost_matrix_l10_1, imagenet_Cost_matrix_l9_0, imagenet_Cost_matrix_l9_1, imagenet_Cost_matrix_l8_0, imagenet_Cost_matrix_l8_1, imagenet_Cost_matrix_l7_0, imagenet_Cost_matrix_l7_1, imagenet_Cost_matrix_l6_0, imagenet_Cost_matrix_l6_1, imagenet_Cost_matrix_l5_0, imagenet_Cost_matrix_l5_1, imagenet_Cost_matrix_l4_0, imagenet_Cost_matrix_l4_1, imagenet_Cost_matrix_l3_0, imagenet_Cost_matrix_l3_1, imagenet_Cost_matrix_l2_0, imagenet_Cost_matrix_l2_1, imagenet_Cost_matrix_l1_0, imagenet_Cost_matrix_l1_1]
    if not os.path.exists('_tiered_imagenet/tiered_imagenet_cost_matrices.pkl'):
        with open('_tiered_imagenet/tiered_imagenet_cost_matrices.pkl', 'wb') as file:
            pickle.dump(imagenet_cost_matrices, file)
    return imagenet_cost_matrices


if __name__ == '__main__':
    # cifar100_cost_matrix()
    distances = imagenet_cost_matrix()
    print(distances)