import torch
import pickle
import numpy as np
from torch.autograd import Variable


def fgvc_get_target_l3(targets):
    save_path = "_fgvc_aircraft/fgvc_aircraft_tree_list_level3.pkl"
    with open(save_path, "rb") as file:
        trees = pickle.load(file)

    manufacturer_target_list = []
    family_target_list = []

    for i in range(targets.size(0)):
        manufacturer_target_list.append(trees[targets[i]][2])
        family_target_list.append(trees[targets[i]][1])

    manufacturer_target_list = Variable(torch.from_numpy(np.array(manufacturer_target_list)).type(torch.LongTensor).cuda())
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)).type(torch.LongTensor).cuda())

    return manufacturer_target_list, family_target_list


def fgvc_map_family_to_variant():
    save_path = "_fgvc_aircraft/fgvc_aircraft_tree_list_level3.pkl"
    with open(save_path, "rb") as file:
        trees = pickle.load(file)
    
    variant_list = []
    for family in np.unique(trees[:, 1]):
        idx = np.where(trees[:,1] == family)[0]
        variant_list.append(list(trees[idx][:, 0]))
    return variant_list


def fgvc_map_manu_to_family():
    save_path = "_fgvc_aircraft/fgvc_aircraft_tree_list_level3.pkl"
    with open(save_path, "rb") as file:
        trees = pickle.load(file)

    family_list = []
    for manu in np.unique(trees[:, 2]):
        idx = np.where(trees[:,2] == manu)[0]
        family_list.append(list(trees[idx][:, 1]))
    return family_list
