import os
import glob
import shutil

import nltk
import platform
import argparse
import requests


from os import makedirs
from os.path import join, exists
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from nltk.tree import Tree

parser = argparse.ArgumentParser()
parser.add_argument("--download-dir", type=str, help="directory to store download fgvc-aircraft files")
parser.add_argument("--dataset-imagefolder", type=str, help="dataset dir suitable for ImageFolder")
parser.add_argument("--construct-tree", action="store_true", help="construct nltk tree")
parser.add_argument("--construct-distance", action="store_true", help="derive hierarchical distance from nltk tree")
preprocessing_opts = parser.parse_args()


def download_fgvc_aircraft_dataset(dest_dir):
    # download dataset
    url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    download_path = os.path.join(dest_dir, "fgvc-aircraft-2013b.tar.gz")

    if not os.path.exists(download_path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(download_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise RuntimeError("ERROR, something went wrong while downloading the fgvc-aircraft dataset")

    # unzip .gz file
    unzip_path = os.path.join(dest_dir, "fgvc-aircraft-2013b.tar")
    if not os.path.exists(unzip_path):
        shutil.unpack_archive(download_path, unzip_path)

    return


def construct_fgvc_aircraft_imagefolder(opts):

    assert opts.dataset_imagefolder is not None

    if exists(opts.dataset_imagefolder):
        raise RuntimeError(f"{opts.dataset_imagefolder} already exists!")
    else:
        makedirs(opts.dataset_imagefolder)

    image_dir = os.path.join(
        opts.download_dir,
        "fgvc-aircraft-2013b.tar",
        "fgvc-aircraft-2013b",
        "data",
        "images",
    )

    annotation_dir = os.path.join(
        opts.download_dir,
        "fgvc-aircraft-2013b.tar",
        "fgvc-aircraft-2013b",
        "data",
    )

    with open(os.path.join(annotation_dir, "images_variant_train.txt")) as f:
        train_labels = list(f)

    for k, line in enumerate(train_labels):
        img_name, img_label = annotation_split(line)
        img_label = f"v-{img_label}"
        src_path = os.path.join(image_dir, f"{img_name}.jpg")
        label_dir = os.path.join(opts.dataset_imagefolder, "train", img_label)
        if not exists(label_dir):
            makedirs(label_dir)
        dest_path = os.path.join(label_dir, f"{img_name}.jpg")
        img = Image.open(src_path)
        col, row = img.size
        # crop out bottom banner
        img = img.crop((0, 0, col, row - 20))
        img.save(dest_path)
        print(f"training set: {k+1}/{len(train_labels)} completed")

    with open(os.path.join(annotation_dir, "images_variant_val.txt")) as f:
        val_labels = list(f)

    for k, line in enumerate(val_labels):
        img_name, img_label = annotation_split(line)
        img_label = f"v-{img_label}"
        src_path = os.path.join(image_dir, f"{img_name}.jpg")
        label_dir = os.path.join(opts.dataset_imagefolder, "val", img_label)
        if not exists(label_dir):
            makedirs(label_dir)
        dest_path = os.path.join(label_dir, f"{img_name}.jpg")
        img = Image.open(src_path)
        col, row = img.size
        # crop out bottom banner
        img = img.crop((0, 0, col, row - 20))
        img.save(dest_path)
        print(f"val set: {k+1}/{len(val_labels)} completed")

    with open(os.path.join(annotation_dir, "images_variant_test.txt")) as f:
        test_labels = list(f)

    for k, line in enumerate(test_labels):
        img_name, img_label = annotation_split(line)
        img_label = f"v-{img_label}"
        src_path = os.path.join(image_dir, f"{img_name}.jpg")
        label_dir = os.path.join(opts.dataset_imagefolder, "test", img_label)
        if not exists(label_dir):
            makedirs(label_dir)
        dest_path = os.path.join(label_dir, f"{img_name}.jpg")
        img = Image.open(src_path)
        col, row = img.size
        # crop out bottom banner
        img = img.crop((0, 0, col, row - 20))
        img.save(dest_path)
        print(f"test set: {k+1}/{len(test_labels)} completed")


def get_filename_list(src_dir, regexp):
    myplatform = platform.system()
    file_name_list = []

    if not os.path.exists(src_dir):
        print(f'Error, directory {src_dir} not exist, force to exit')
        exit()

    if myplatform == 'Windows':
        split_char = '\\'
    else:
        split_char = '/'

    for file_path in glob.glob(os.path.join(src_dir, regexp)):
        filename = file_path.split(split_char)
        file_name_list.append(filename[-1])

    return file_name_list


def annotation_split(line):
    if line[-1] == '\n':
        line = line[:-1] # get rid of '\n'
    tokens = line.split(' ')
    img_name = tokens[0]
    label = ' '.join(tokens[1:])
    return img_name, label


def pretty_print_nltk_tree(tree, tab_len=13):
    print(pretty_print_recursive(tree, 0, '', tab_len))
    return


def pretty_print_recursive(tree, depth, printstr, tab_len=13):
    tab = '|' + ' ' * (tab_len - 1)
    tab2 = '-' + ' ' * (tab_len - 1)

    if not isinstance(tree, str):  # not a leaf node
        word = tree.label()
    else:
        word = tree  # leaf node is represented as str

    # maintain label length to tab_len
    if len(word) > tab_len:
        print(f'truncate node label {word} to {word[:tab_len]}')
        word = word[:tab_len]

    while len(word) < tab_len:
        word = word + ' '

    for i in range(depth - 1):
        printstr = printstr + tab

    if depth > 0:
        printstr = printstr + tab2

    printstr = printstr + word
    printstr = printstr + '\n'

    if not isinstance(tree, str):
        for node in tree:
            printstr = pretty_print_recursive(node, depth + 1, printstr, tab_len)

    return printstr


def construct_fgvc_aircraft_nltk_tree(manu, family, variant):
    """
    manu: defaultdict(set)
    family: defaultdict(set)
    variant: set
    """
    # dictionary of nltk tree nodes
    node_dict = {}

    # variant level
    for n in variant:
        node_dict[n] = n

    # family level
    for k, v in family.items():
        children_list = []
        for child in v:
            children_list.append(node_dict[child])
        node_dict[k] = Tree(k, children_list)

    # manufacturer level
    for k, v in manu.items():
        children_list = []
        for child in v:
            children_list.append(node_dict[child])
        node_dict[k] = Tree(k, children_list)

    # artifectial root
    children_list = []
    for child in manu.keys():
        children_list.append(node_dict[child])
    node_dict['unknown'] = Tree('unknown', children_list)
    return node_dict['unknown']


def root_to_leaf_path(tree, leaf):
    if type(tree) is nltk.Tree:
        path = []
        for node in tree:
            # recursion
            path = root_to_leaf_path(node, leaf)
            if len(path) != 0:
                break

        if len(path) != 0:
            path = [tree] + path
        else:
            path = []

    elif tree == leaf:  # base case 0
        path = [tree]

    else:  # base case 1
        path = []

    return path


def lowest_common_ancestor(tree, leaf_a, leaf_b):
    path_a = root_to_leaf_path(tree, leaf_a)
    # print(f"path a:\n{path_a}")
    path_b = root_to_leaf_path(tree, leaf_b)
    # print(f"path b:\n{path_b}")
    max_depth = min(len(path_a), len(path_b))

    cnt = 0
    for i in range(max_depth):
        if path_a[i] == path_b[i]:
            cnt += 1
        else:
            break
    return path_a[cnt - 1]


def hierarchical_distance(tree, leaf_a, leaf_b):
    lca = lowest_common_ancestor(tree, leaf_a, leaf_b)
    return lca.height() - 1


def tree_to_distance_dict(tree):
    leaves = tree.leaves()
    n_leaves = len(leaves)
    distance_dict = {}
    for i in range(n_leaves-1):  # i in [0, n_leaves-1)
        distance_dict[(leaves[i], leaves[i])] = 0
        for j in range(i+1, n_leaves):  # j in [i+1, n_leaves]
            hdistance = hierarchical_distance(tree, leaves[i], leaves[j])
            distance_dict[(leaves[i], leaves[j])] = hdistance
            distance_dict[(leaves[j], leaves[i])] = hdistance
    distance_dict[(leaves[-1], leaves[-1])] = 0
    return distance_dict


def main(opts):

    # download dataset
    download_fgvc_aircraft_dataset(opts.download_dir)
    data_dir = os.path.join(opts.download_dir, "fgvc-aircraft-2013b.tar", "fgvc-aircraft-2013b", "data")

    # for constructing fgvc-aircraft image folder
    construct_fgvc_aircraft_imagefolder(opts)

    if opts.construct_tree:
        variant_path = os.path.join(data_dir, "images_variant_train.txt")
        family_path = os.path.join(data_dir, "images_family_train.txt")
        manufacturer_path = os.path.join(data_dir, "images_manufacturer_train.txt")

        # read variant, each line is ending with '\n'
        with open(variant_path) as f:
            variant_lines = list(f)

        # read family
        with open(family_path) as f:
            family_lines = list(f)

        # read manufacturer
        with open(manufacturer_path) as f:
            manufacturer_lines = list(f)

        manu = defaultdict(set)
        family = defaultdict(set)
        variant = set()

        # collect all the manufacturer, family, and variant
        for k in tqdm(range(len(variant_lines))):
            img_name, variant_label = annotation_split(variant_lines[k])
            _, family_label = annotation_split(family_lines[k])
            _, manu_label = annotation_split(manufacturer_lines[k])

            # print(f"{img_name}: {variant_label} is {family_label} of {manu_label}")
            # break

            # add tag to variant_label, family_label, manu_label -> duplicate names between different levels
            variant_label = f'v-{variant_label}'
            family_label = f'f-{family_label}'
            manu_label = f'm-{manu_label}'

            manu[manu_label].add(family_label)
            family[family_label].add(variant_label)
            variant.add(variant_label)

        # one can save the tree and distance files as pickles here
        tree = construct_fgvc_aircraft_nltk_tree(manu, family, variant)
        pretty_print_nltk_tree(tree, 30)

    if opts.construct_tree and opts.construct_distance:
        distance = tree_to_distance_dict(tree)
        cnt = 5
        print("preview of distance file")
        for i, (k, v) in enumerate(distance.items()):
            print(f"distance[{k}]: {v}")
            if i > cnt:
                break


if __name__ == "__main__":
    main(preprocessing_opts)



