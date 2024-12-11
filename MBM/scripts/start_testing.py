import argparse
import os
import json


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.optim
import numpy as np

from MBM.better_mistakes.model.init import init_model_on_gpu
from MBM.better_mistakes.model.run_xent import run

from MBM.better_mistakes.util.config import load_config
from MBM.better_mistakes.trees import load_distances
from util import data_loader, logger

# HAFrame
from HAFrame.losses import MixedLoss_CEandGeneralSCSL
from HAFrame.distance import distance_dict_to_mat
from HAFrame.solve_HAF import hdistance_to_similarity_matrix


DATASET_NAMES = [ "tiered-imagenet-224", "inaturalist19-224", "cifar-100", "fgvc-aircraft"]
LOSS_NAMES = [
    "cross-entropy", "flamingo-l3", "flamingo-l5", "flamingo-l7", "flamingo-l12",
    "hafeat-l3-cejsd-wtconst-dissim", "hafeat-l5-cejsd-wtconst-dissim",
    "hafeat-l7-cejsd-wtconst-dissim", "hafeat-l12-cejsd-wtconst-dissim",
    "mixed-ce-gscsl",
    "ours_l3","ours_l5","ours_l7","ours_l12", 
]


def main(test_opts):
    gpus_per_node = torch.cuda.device_count()

    assert test_opts.out_folder

    if test_opts.start:
        opts = test_opts
        opts.epochs = 0
    else:
        expm_json_path = os.path.join(test_opts.out_folder, "opts.json")
        assert os.path.isfile(expm_json_path)
        with open(expm_json_path) as fp:
            opts = json.load(fp)
            # convert dictionary to namespace
            opts = argparse.Namespace(**opts)
            opts.out_folder = None
            opts.epochs = 0

        if test_opts.data_path is None or opts.data_path is None:
            opts.data_paths = load_config(test_opts.data_paths_config)
            opts.data_path = opts.data_paths[opts.data]

        opts.start = test_opts.start # to update value of start if testing

    # Setup data loaders --------------------------------------------------------------------------------------------- #
    if opts.start == "testing":
        test_dataset, test_loader = data_loader.test_data_loader(opts)
    elif opts.start == "validating":
        _, test_dataset, _, test_loader = data_loader.train_data_loader(opts)
    else:
        raise ValueError(f"unrecognized opts.start option: {opts.start}")

    # Load hierarchy and classes ------------------------------------------------------------------------------------- #
    if opts.data == "fgvc-aircraft":
        distances = load_distances(opts.data, 'original', opts.data_dir)
    else:
        distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)

    if opts.data == "cifar-100":
        classes = test_dataset.class_to_idx
        classes = ["L5-" + str(classes[i]) for i in classes]
    else:
        classes = test_dataset.classes

    opts.num_classes = len(classes)

    # carry class string labels with opts
    opts.class_str_labels = classes

    # Model, loss, optimizer ----------------------------------------------------------------------------------------- #
    # setup loss
    if opts.loss == "cross-entropy":
        loss_function = nn.CrossEntropyLoss().cuda(opts.gpu)
    elif opts.loss == "mixed-ce-gscsl":
        assert (opts.haf_gamma is not None) and (opts.loss_schedule is not None)
        loss_function = MixedLoss_CEandGeneralSCSL(opts.num_classes).cuda(opts.gpu)
    elif opts.loss in LOSS_NAMES:
        loss_function = None
    else:
        raise RuntimeError("Unkown loss {}".format(opts.loss))

    corrector = lambda x: x

    # create the solft labels
    if opts.loss == "mixed-ce-gscsl":
        # cosine similarities derived from ideal collapse to HAFrame
        distance_matrix = distance_dict_to_mat(distances, classes)
        # pair-wise cosine similarity matrix as soft_labels
        soft_labels = hdistance_to_similarity_matrix(distance_matrix, opts.haf_gamma, len(classes))
        soft_labels = torch.Tensor(soft_labels)
    else:
        soft_labels = None

    # Test ----------------------------------------------------------------------------------------------------------- #
    summaries, summaries_table = dict(), dict()

    # setup model
    model = init_model_on_gpu(gpus_per_node, opts, distances)

    if opts.checkpoint_path is None:
        checkpoint_id = "best.checkpoint.pth.tar"
        checkpoint_path = os.path.join(test_opts.out_folder, checkpoint_id)
    else:
        checkpoint_path = opts.checkpoint_path

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        # logger._print("=> loaded checkpoint '{}'".format(checkpoint_path), os.path.join(test_opts.out_folder, logs_txt_name))
        opts.logger_val.info("=> loaded checkpoint '{}'".format(checkpoint_path))
    else:
        # logger._print("=> no checkpoint found at '{}'".format(checkpoint_path), os.path.join(test_opts.out_folder, logs_txt_name))
        opts.logger_val.info("=> no checkpoint found at '{}'".format(checkpoint_path))
        raise RuntimeError

    summary, _ = run(test_loader, model, loss_function, distances, soft_labels, classes, opts, 0, 0, is_inference=True, corrector=corrector)

    for k in summary.keys():
        val = summary[k]
        if k not in summaries:
            summaries[k] = []
        summaries[k].append(val)

    for k in summaries.keys():
        avg = np.mean(summaries[k])
        conf95 = 1.96 * np.std(summaries[k]) / np.sqrt(len(summaries[k]))
        summaries_table[k] = (avg, conf95)
        # logger._print("\t\t\t\t%20s: %.2f" % (k, summaries_table[k][0]) + " +/- %.4f" % summaries_table[k][1], os.path.join(test_opts.out_folder, logs_txt_name))
        opts.logger_val.info("\t\t\t\t%20s: %.2f" % (k, summaries_table[k][0]) + " +/- %.4f" % summaries_table[k][1])

    summary_name = "test_summary" if opts.start == "testing" else "validation_summary"

    with open(os.path.join(test_opts.out_folder, f"{summary_name}.json"), "w") as fp:
        json.dump(summaries, fp, indent=4)
    with open(os.path.join(test_opts.out_folder, f"{summary_name}_table.json"), "w") as fp:
        json.dump(summaries_table, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_folder", help="Path to data paths yaml file", default=None)
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="../../data_paths.yml")
    parser.add_argument("--data-path", default=None, help="explicit loc of the data folder if None use config file.")
    parser.add_argument("--data_dir", default="../../data/", help="Folder containing the supplementary data")
    parser.add_argument("--workers", default=2, type=int, help="number of data loading workers")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    opts = parser.parse_args()

    main(opts)
