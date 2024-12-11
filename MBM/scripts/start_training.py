import argparse
import os
import json

import numpy as np
from distutils.util import strtobool as boolean
from pprint import PrettyPrinter

import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from MBM.better_mistakes.util.rand import make_deterministic
from MBM.better_mistakes.util.folders import get_expm_folder

from MBM.better_mistakes.util.config import load_config
from MBM.better_mistakes.model.init import init_model_on_gpu
from MBM.better_mistakes.model.run_xent import run
from MBM.better_mistakes.trees import load_distances
from util import data_loader, logger

# HAFrame
from HAFrame.losses import MixedLoss_CEandGeneralSCSL
from HAFrame.distance import distance_dict_to_mat
from HAFrame.solve_HAF import hdistance_to_similarity_matrix

CUSTOM_MODELS = ["custom_resnet50", "haframe_resnet50", "wide_resnet", "haframe_wide_resnet"]
MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
MODEL_NAMES.extend(CUSTOM_MODELS)
LOSS_NAMES = [
    "cross-entropy", "flamingo-l3", "flamingo-l5", "flamingo-l7", "flamingo-l12",
    "hafeat-l3-cejsd-wtconst-dissim", "hafeat-l5-cejsd-wtconst-dissim",
    "hafeat-l7-cejsd-wtconst-dissim", "hafeat-l12-cejsd-wtconst-dissim",
    "mixed-ce-gscsl",
    "ours_l3", "ours_l5", "ours_l7", "ours_l12"
]

OPTIMIZER_NAMES = ["adagrad", "adam", "adam_amsgrad", "rmsprop", "SGD", "custom_sgd"]
DATASET_NAMES = [ "tiered-imagenet-224", "inaturalist19-224", "cifar-100", "fgvc-aircraft"]


def cosine_anneal_schedule(t, nb_epoch, lower_bound="ignored"):
    cos_inner = np.pi * (t % nb_epoch)  # t - 1 is used when t has 1-based indexing.
    cos_inner /= nb_epoch
    cos_out = np.cos(cos_inner) + 1
    return float( 1 / 2 * cos_out)


def cosine_anneal_schedule_raise_lower_bound(t, nb_epoch, lower_bound=1e-3):
    cos_inner = np.pi * (t % nb_epoch)  # t - 1 is used when t has 1-based indexing.
    cos_inner /= nb_epoch
    cos_out = np.cos(cos_inner) + 1
    # the returned lr will be divided by 10 as backbone learning rate
    # lower_bound 1e-3 / 10 -> 1e-4 for backbone,
    # and 1e-3 for classifiers and transformation module (if not included in backbone)
    return float( 1 / 2 * cos_out) + lower_bound

def step_schedule(t, nb_epoch):
    factor = nb_epoch / 200
    epoch = t
    if epoch <= 60 * factor:
        step_out = 1.
    elif epoch <= 120 * factor:
        step_out = 0.2
    elif epoch <= 160 * factor:
        step_out = 0.04
    else:
        step_out = 0.008
    return step_out

def step_wise_schedule(opts, optimizer, milestones):
    milestones = [int(opts.epochs * ms) for ms in milestones]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.2,
        last_epoch=-1
    )
    return lr_scheduler


def main_worker(gpus_per_node, opts):
    # Worker setup
    if opts.gpu is not None:
        print("Use GPU: {} for training".format(opts.gpu))

    # Enables the cudnn auto-tuner to find the best algorithm to use for your hardware
    cudnn.benchmark = True

    # pretty printer for cmd line options
    pp = PrettyPrinter(indent=4)

    # Setup data loaders --------------------------------------------------------------------------------------------- #
    train_dataset, val_dataset, train_loader, val_loader = data_loader.train_data_loader(opts)

    # Load hierarchy and classes ------------------------------------------------------------------------------------- #
    if opts.data == "fgvc-aircraft":
        distances = load_distances(opts.data, 'original', opts.data_dir)
    else:
        distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)

    if opts.data == "cifar-100":
        classes = train_dataset.class_to_idx
        classes = ["L5-" + str(classes[i]) for i in classes]
    else:
        classes = train_dataset.classes

    # Adjust the number of epochs to the size of the dataset
    num_batches = len(train_loader)

    if opts.epochs is None:
        opts.epochs = int(round(opts.num_training_steps / num_batches))

    opts.num_classes = len(classes)
    print("num_classes: ", opts.num_classes)

    # carry class string labels with opts
    opts.class_str_labels = classes

    # Model, loss, optimizer ----------------------------------------------------------------------------------------- #

    # setup model
    model = init_model_on_gpu(gpus_per_node, opts, distances)

    # setup optimizer
    optimizer = _select_optimizer(model, opts)

    # load from checkpoint if exists
    steps = _load_checkpoint(opts, model, optimizer)

    # default cosine annealing scheduler
    if opts.raise_lower_bound:
        lr_scheduler = cosine_anneal_schedule_raise_lower_bound
    else:
        lr_scheduler = cosine_anneal_schedule
    if opts.optimizer == "custom_sgd" and opts.data == "cifar-100":
        lr_scheduler = step_schedule

    if opts.optimizer == "step_sgd" :
        lr_scheduler = step_wise_schedule(opts, optimizer, [1.0/2.0, 3.0/4.0])

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

    # create the solft labels
    if opts.loss == "mixed-ce-gscsl":
        # cosine similarities derived from ideal collapse onto HAFrame
        distance_matrix = distance_dict_to_mat(distances, classes)
        # pair-wise cosine similarity matrix as soft_labels
        soft_labels = hdistance_to_similarity_matrix(distance_matrix, opts.haf_gamma, len(classes))
        soft_labels = torch.Tensor(soft_labels)
    else:
        soft_labels = None

    corrector = lambda x: x

    # Training/evaluation -------------------------------------------------------------------------------------------- #
    best_accuracy = 0
    opts.tb_writer = SummaryWriter(os.path.join(opts.out_folder, "tb"))
    for epoch in range(opts.start_epoch, opts.epochs):
        # do we validate at this epoch?
        do_validate = epoch % opts.val_freq == 0

        if opts.data == "inaturalist19-224" and opts.optimizer == "custom_sgd":
            if opts.loss in ["cross-entropy", "mixed-ce-gscsl"]: 
                if opts.arch in ["custom_resnet50", "haframe_resnet50"]:
                    if not opts.larger_backbone:
                        optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                        optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                        optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                    else:
                        optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                        optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                        optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
            elif opts.loss in ["flamingo-l7", "hafeat-l7-cejsd-wtconst-dissim"]:
                if not opts.larger_backbone:
                    optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[3]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[4]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[5]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[6]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[7]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[8]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                else:
                    optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[3]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[4]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[5]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[6]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[7]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                    optimizer.param_groups[8]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
            elif opts.loss in ["ours_l7"]:
                if not opts.larger_backbone:
                    optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[3]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[4]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[5]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[6]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[7]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[8]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[9]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[10]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[11]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[12]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[13]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[14]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[15]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                else:
                    optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[3]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[4]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[5]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[6]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[7]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[8]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[9]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[10]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[11]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[12]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[13]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[14]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                    optimizer.param_groups[15]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10

        if opts.data == "tiered-imagenet-224" and opts.optimizer == "custom_sgd": 
            if opts.loss in ["cross-entropy", "mixed-ce-gscsl"]:
                if opts.arch in ["custom_resnet50", "haframe_resnet50"]:
                    if not opts.larger_backbone:
                        optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                        optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                        optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                    else:
                        optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                        optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                        optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
            elif opts.loss in ["flamingo-l12", "hafeat-l12-cejsd-wtconst-dissim"]:
                if not opts.larger_backbone:
                    optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[3]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[4]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[5]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[6]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[7]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[8]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[9]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[10]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[11]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[12]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[13]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                else:
                    optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[3]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[4]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[5]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[6]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[7]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[8]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[9]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[10]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[11]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[12]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                    optimizer.param_groups[13]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
            elif opts.loss in ["ours_l12"]:
                if not opts.larger_backbone:
                    optimizer.param_groups[0]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[2]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[3]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[4]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[5]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[6]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[7]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[8]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[9]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[10]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[11]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[12]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[13]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[14]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[15]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[16]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[17]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[18]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[19]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[20]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[21]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[22]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[23]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[24]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[25]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                else:
                    optimizer.param_groups[0]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[2]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[3]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[4]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[5]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[6]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[7]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[8]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[9]['lr']  = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[10]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[11]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[12]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[13]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[14]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[15]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[16]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[17]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[18]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[19]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[20]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[21]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[22]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[23]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[24]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                    optimizer.param_groups[25]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs) / 10

        if opts.data == "cifar-100" and opts.optimizer == "custom_sgd":
            if opts.arch in ["wide_resnet", "haframe_wide_resnet"]:
                if opts.loss in ["cross-entropy", "mixed-ce-gscsl"]:
                    optimizer.param_groups[0]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                if opts.loss in ["hafeat-l5-cejsd-wtconst-dissim", "flamingo-l5"]:
                    optimizer.param_groups[0]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 1
                    optimizer.param_groups[1]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 2
                    optimizer.param_groups[2]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 3
                    optimizer.param_groups[3]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 3
                    optimizer.param_groups[4]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 3
                    optimizer.param_groups[5]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs) / 10  # backbone
                if opts.loss in ["ours_l5"]:
                    optimizer.param_groups[0]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 1_0
                    optimizer.param_groups[1]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 1_1
                    optimizer.param_groups[2]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 2_0
                    optimizer.param_groups[3]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 2_1
                    optimizer.param_groups[4]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 3_0
                    optimizer.param_groups[5]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 3_1
                    optimizer.param_groups[6]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 4_0
                    optimizer.param_groups[7]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 4_1
                    optimizer.param_groups[8]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 5_0
                    optimizer.param_groups[9]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs)       # level 5_1
                    optimizer.param_groups[10]['lr'] = opts.lr * lr_scheduler(epoch, opts.epochs) / 10  # backbone

        if opts.data == "fgvc-aircraft" and opts.optimizer == "custom_sgd":
            if opts.loss in ["cross-entropy", "mixed-ce-gscsl"]:
                if opts.arch in ["custom_resnet50", "haframe_resnet50"]:
                    if not opts.larger_backbone:
                        optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                        optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                        optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                    else:
                        optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)
                        optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10
                        optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10

            elif opts.loss in ["flamingo-l3", "hafeat-l3-cejsd-wtconst-dissim"]:
                if not opts.larger_backbone:
                    optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 3
                    optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 2
                    optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 1
                    optimizer.param_groups[3]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # mapping
                    optimizer.param_groups[4]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10  # backbone
                else:
                    optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 3
                    optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 2
                    optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 1
                    optimizer.param_groups[3]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10  # mapping
                    optimizer.param_groups[4]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10  # backbone
            elif opts.loss in ["ours_l3"]:
                if not opts.larger_backbone:
                    optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 3
                    optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 3
                    optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 2
                    optimizer.param_groups[3]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 2
                    optimizer.param_groups[4]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 1
                    optimizer.param_groups[5]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 1
                    optimizer.param_groups[6]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # mapping
                    optimizer.param_groups[7]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10  # backbone
                else:
                    optimizer.param_groups[0]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 3
                    optimizer.param_groups[1]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 3
                    optimizer.param_groups[2]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 2
                    optimizer.param_groups[3]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 2
                    optimizer.param_groups[4]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 1
                    optimizer.param_groups[5]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs)       # cls 1
                    optimizer.param_groups[6]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10  # mapping
                    optimizer.param_groups[7]['lr'] =  opts.lr * lr_scheduler(epoch, opts.epochs) / 10  # backbone

        # training for one epoch
        summary_train, steps = run(
            train_loader, model, loss_function, distances,
            soft_labels, classes, opts, epoch, steps, optimizer, is_inference=False, corrector=corrector,
        )

        if epoch == opts.start_epoch:
            title = 'Epoch\t'
            for k, v in summary_train.items():
                title += (k + '\t')
            opts.logger_train.info(title)
        log_line = str(epoch) + '\t'
        for k,v in summary_train.items():
            log_line += f'{v:.4f}' + '\t'
        opts.logger_train.info(log_line)

        if opts.optimizer == "step_sgd":
            lr_scheduler.step()
            print(f"updated learning rate: {optimizer.param_groups[0]['lr']}")

        # print summary of the epoch and save checkpoint
        state = {
            "epoch": epoch + 1, "steps": steps, "arch": opts.arch,
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()
        }

        _save_checkpoint(state, opts.out_folder)
        print("\nSummary for epoch %04d (for train set):" % epoch)
        pp.pprint(summary_train)

        # if periodic checkpointing is turned on
        if (epoch > 0)  and (opts.ckpt_freq != 0) and (epoch % opts.ckpt_freq == 0):
            _save_checkpoint(state, os.path.join(opts.out_folder, "checkpoints"), epoch)

        # validation
        if do_validate:
            summary_val, steps = run(
                val_loader, model, loss_function, distances, soft_labels,
                classes, opts, epoch, steps, is_inference=True, corrector=corrector,
            )

            if epoch == opts.start_epoch:
                title = 'Epoch\t'
                for k, v in summary_val.items():
                    title += (k + '\t')
                opts.logger_val.info(title)
            log_line = str(epoch) + '\t'
            for k,v in summary_val.items():
                log_line += f'{v:.4f}' + '\t'
            opts.logger_val.info(log_line)

            if summary_val["accuracy_top/01"] > best_accuracy:
                best_accuracy = summary_val["accuracy_top/01"]
                print(f"Best accuracy @{epoch} is {best_accuracy}%")
                _save_best_checkpoint(state, epoch, opts.out_folder)
            if (opts.loss == "cross-entropy") and (epoch == 0 or epoch == 65):
                _save_checkpoint(state, opts.out_folder, epoch=epoch)
            print("\nSummary for epoch %04d (for val set):" % epoch)
            pp.pprint(summary_val)
            print("\n\n")


def _load_checkpoint(opts, model, optimizer):
    if os.path.isfile(os.path.join(opts.out_folder, "checkpoint.pth.tar")):
        print("=> loading checkpoint '{}'".format(opts.out_folder))
        checkpoint = torch.load(os.path.join(opts.out_folder, "checkpoint.pth.tar"))
        opts.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        steps = checkpoint["steps"]
        print("=> loaded checkpoint '{}' (epoch {})".format(opts.out_folder, checkpoint["epoch"]))
    elif opts.pretrained_folder is not None:
        # import pdb; pdb.set_trace()
        if os.path.exists(opts.pretrained_folder):
            print("=> loading pretrained checkpoint '{}'".format(opts.pretrained_folder))
            if os.path.isdir(opts.pretrained_folder):
                checkpoint = torch.load(os.path.join(opts.pretrained_folder, "checkpoint.pth.tar"))
            else:
                checkpoint = torch.load(opts.pretrained_folder)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            steps = 0
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(opts.pretrained_folder, checkpoint["epoch"]))
        else:
            raise FileNotFoundError("Can not find {}".format(opts.pretrained_folder))
    else:
        steps = 0
        print("=> no checkpoint found at '{}'".format(opts.out_folder))

    return steps


def _save_checkpoint(state, out_folder, epoch=None):
    filename = os.path.join(out_folder, "checkpoint.pth.tar")
    if epoch is not None:
        filename = os.path.join(out_folder, "epoch_" + str(epoch) + "_checkpoint.pth.tar")
    torch.save(state, filename)


def _save_best_checkpoint(state, epoch, out_folder):
    filename = os.path.join(out_folder, "best.checkpoint.pth.tar")
    torch.save(state, filename)


def _adam_amsgrad(opts, model):
    return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=True, )


def _custom_sgd(opts, model):

    if opts.data == "cifar-100":
        if opts.loss in ["cross-entropy", "mixed-ce-gscsl"]:
            if opts.arch in ["wide_resnet", "haframe_wide_resnet"]:
                return torch.optim.SGD([
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
        if opts.loss in ["flamingo-l5", "hafeat-l5-cejsd-wtconst-dissim"]:
            if opts.arch in ["wide_resnet", "haframe_wide_resnet"]:
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.classifier_4.parameters(), 'lr': 0.1},
                    {'params': model.classifier_5.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
        if opts.loss in ["ours_l5"]:
            if opts.arch in ["wide_resnet", "haframe_wide_resnet"]:
                return torch.optim.SGD([
                    {'params': model.classifier_1_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_1_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_4_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_4_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_5_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_5_1.parameters(), 'lr': opts.lr},
                    {'params': model.features_2.parameters(), 'lr': opts.lr/10},
                ], momentum=0.9, weight_decay=5e-4)

    if opts.data == "inaturalist19-224":
        if opts.loss in ["cross-entropy", "mixed-ce-gscsl"]:
            if opts.arch in ["custom_resnet50", "haframe_resnet50"]:
                if not opts.larger_backbone:
                    return torch.optim.SGD([
                        {'params': model.classifier_3.parameters(), 'lr': 0.1},
                        {'params': model.features_1.parameters(), 'lr': 0.1},
                        {'params': model.features_2.parameters(), 'lr': 0.01},
                    ], momentum=0.9, weight_decay=5e-4)
                else:
                    return torch.optim.SGD([
                        {'params': model.classifier_3.parameters(), 'lr': 0.1},
                        {'params': model.features_1.parameters(), 'lr': 0.01}, # transformation module
                        {'params': model.features_2.parameters(), 'lr': 0.01},
                    ], momentum=0.9, weight_decay=5e-4)
        if opts.loss in ["flamingo-l7", "hafeat-l7-cejsd-wtconst-dissim"]:
            if not opts.larger_backbone:
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.classifier_4.parameters(), 'lr': 0.1},
                    {'params': model.classifier_5.parameters(), 'lr': 0.1},
                    {'params': model.classifier_6.parameters(), 'lr': 0.1},
                    {'params': model.classifier_7.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
            else:
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.classifier_4.parameters(), 'lr': 0.1},
                    {'params': model.classifier_5.parameters(), 'lr': 0.1},
                    {'params': model.classifier_6.parameters(), 'lr': 0.1},
                    {'params': model.classifier_7.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.01},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
        if opts.loss in ["ours_l7"]:
            if not opts.larger_backbone:
                return torch.optim.SGD([
                    {'params': model.classifier_1_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_1_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_4_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_4_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_5_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_5_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_6_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_6_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_7_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_7_1.parameters(), 'lr': opts.lr},
                    {'params': model.features_1.parameters(), 'lr': opts.lr},
                    {'params': model.features_2.parameters(), 'lr': opts.lr/10},
                ], momentum=0.9, weight_decay=5e-4)
            else:
                return torch.optim.SGD([
                    {'params': model.classifier_1_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_1_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_4_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_4_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_5_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_5_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_6_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_6_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_7_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_7_1.parameters(), 'lr': opts.lr},
                    {'params': model.features_1.parameters(), 'lr': opts.lr/10},
                    {'params': model.features_2.parameters(), 'lr': opts.lr/10},
                ], momentum=0.9, weight_decay=5e-4)

    if opts.data == "tiered-imagenet-224":
        if opts.loss in ["cross-entropy", "mixed-ce-gscsl"]:
            if opts.arch in ["custom_resnet50", "haframe_resnet50"]:
                if not opts.larger_backbone:
                    return torch.optim.SGD([
                        {'params': model.classifier_3.parameters(), 'lr': 0.1},
                        {'params': model.features_1.parameters(), 'lr': 0.1},
                        {'params': model.features_2.parameters(), 'lr': 0.01},
                    ], momentum=0.9, weight_decay=5e-4)
                else:
                    return torch.optim.SGD([
                        {'params': model.classifier_3.parameters(), 'lr': 0.1},
                        {'params': model.features_1.parameters(), 'lr': 0.01},
                        {'params': model.features_2.parameters(), 'lr': 0.01},
                    ], momentum=0.9, weight_decay=5e-4)
        if opts.loss in ["flamingo-l12", "hafeat-l12-cejsd-wtconst-dissim"]:
            if not opts.larger_backbone:
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.classifier_4.parameters(), 'lr': 0.1},
                    {'params': model.classifier_5.parameters(), 'lr': 0.1},
                    {'params': model.classifier_6.parameters(), 'lr': 0.1},
                    {'params': model.classifier_7.parameters(), 'lr': 0.1},
                    {'params': model.classifier_8.parameters(), 'lr': 0.1},
                    {'params': model.classifier_9.parameters(), 'lr': 0.1},
                    {'params': model.classifier_10.parameters(), 'lr': 0.1},
                    {'params': model.classifier_11.parameters(), 'lr': 0.1},
                    {'params': model.classifier_12.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
            else:
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.classifier_4.parameters(), 'lr': 0.1},
                    {'params': model.classifier_5.parameters(), 'lr': 0.1},
                    {'params': model.classifier_6.parameters(), 'lr': 0.1},
                    {'params': model.classifier_7.parameters(), 'lr': 0.1},
                    {'params': model.classifier_8.parameters(), 'lr': 0.1},
                    {'params': model.classifier_9.parameters(), 'lr': 0.1},
                    {'params': model.classifier_10.parameters(), 'lr': 0.1},
                    {'params': model.classifier_11.parameters(), 'lr': 0.1},
                    {'params': model.classifier_12.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.01},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
        if opts.loss in ["ours_l12"]:
            if not opts.larger_backbone:
                return torch.optim.SGD([
                    {'params': model.classifier_1_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_1_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_4_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_4_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_5_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_5_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_6_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_6_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_7_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_7_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_8_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_8_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_9_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_9_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_10_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_10_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_11_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_11_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_12_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_12_1.parameters(), 'lr': opts.lr},
                    {'params': model.features_1.parameters(), 'lr': opts.lr},
                    {'params': model.features_2.parameters(), 'lr': opts.lr/10},
                ], momentum=0.9, weight_decay=5e-4)
            else:
                return torch.optim.SGD([
                    {'params': model.classifier_1_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_1_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_4_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_4_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_5_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_5_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_6_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_6_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_7_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_7_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_8_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_8_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_9_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_9_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_10_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_10_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_11_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_11_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_12_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_12_1.parameters(), 'lr': opts.lr},
                    {'params': model.features_1.parameters(), 'lr': opts.lr/10},
                    {'params': model.features_2.parameters(), 'lr': opts.lr/10},
                ], momentum=0.9, weight_decay=5e-4)

    if opts.data == "fgvc-aircraft":
        if opts.loss in ["cross-entropy", "mixed-ce-gscsl"]:
            if opts.arch in ["custom_resnet50", "haframe_resnet50"]:
                if not opts.larger_backbone:
                    return torch.optim.SGD([
                        {'params': model.classifier_3.parameters(), 'lr': 0.1},
                        {'params': model.features_1.parameters(), 'lr': 0.1},
                        {'params': model.features_2.parameters(), 'lr': 0.01},
                    ], momentum=0.9, weight_decay=5e-4)
                else:
                    return torch.optim.SGD([
                        {'params': model.classifier_3.parameters(), 'lr': 0.1},
                        {'params': model.features_1.parameters(), 'lr': 0.01},  # transformation module
                        {'params': model.features_2.parameters(), 'lr': 0.01},
                    ], momentum=0.9, weight_decay=5e-4)
        if opts.loss in ["flamingo-l3", "hafeat-l3-cejsd-wtconst-dissim"]:
            if not opts.larger_backbone:
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
            else:
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.01},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
        if opts.loss in ["ours_l3"]:
            if not opts.larger_backbone:
                return torch.optim.SGD([
                    {'params': model.classifier_1_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_1_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_1.parameters(), 'lr': opts.lr},
                    {'params': model.features_1.parameters(), 'lr': opts.lr},
                    {'params': model.features_2.parameters(), 'lr': opts.lr/10},
                ], momentum=0.9, weight_decay=5e-4)
            else:
                return torch.optim.SGD([
                    {'params': model.classifier_1_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_1_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_2_1.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_0.parameters(), 'lr': opts.lr},
                    {'params': model.classifier_3_1.parameters(), 'lr': opts.lr},
                    {'params': model.features_1.parameters(), 'lr': opts.lr/10},
                    {'params': model.features_2.parameters(), 'lr': opts.lr/10},
                ], momentum=0.9, weight_decay=5e-4)

    return None


def _step_sgd(opts, model):
    decay_parameters = []
    zero_decay_parameters = []
    no_gradient_parameters = []

    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'bn' in n:
                zero_decay_parameters.append(n)
            elif 'prelu' in n:
                zero_decay_parameters.append(n)
            elif 'act_func' in n:
                zero_decay_parameters.append(n)
            else:
                decay_parameters.append(n)
        else:
            no_gradient_parameters.append(n)
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if n in decay_parameters],
            'weight_decay': 1e-4,
        },
        {
            'params': [p for n, p in model.named_parameters() if n in zero_decay_parameters],
            'weight_decay': 0.0,
        }
    ]

    return torch.optim.SGD(optimizer_grouped_parameters, lr=opts.lr, momentum=0.9)


def _select_optimizer(model, opts):
    if opts.optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=False)
    elif opts.optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0)
    elif opts.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0, nesterov=False)
    elif opts.optimizer == "adam_amsgrad":
        return _adam_amsgrad(opts, model)
    elif opts.optimizer == "custom_sgd":
        return _custom_sgd(opts, model)
    elif opts.optimizer == "step_sgd":
        return _step_sgd(opts, model)
    else:
        raise ValueError("Unknown optimizer", opts.optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18", choices=MODEL_NAMES,
                        help="model architecture: | ".join(MODEL_NAMES))
    parser.add_argument("--loss", default="cross-entropy", choices=LOSS_NAMES, help="loss type: | ".join(LOSS_NAMES))
    parser.add_argument("--optimizer", default="adam_amsgrad", choices=OPTIMIZER_NAMES,
                        help="loss type: | ".join(OPTIMIZER_NAMES))
    parser.add_argument("--lr", default=1e-5, type=float, help="initial learning rate of optimizer")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay of optimizer")
    parser.add_argument("--pretrained", type=boolean, default=True, help="start from ilsvrc12/imagenet model weights")
    parser.add_argument("--pretrained_folder", type=str, default=None,
                        help="folder or file from which to load the network weights")
    parser.add_argument("--dropout", default=0.0, type=float, help="Prob of dropout for network FC layer")
    parser.add_argument("--data_augmentation", type=boolean, default=True, help="Train with basic data augmentation")
    parser.add_argument("--num_training_steps", default=200000, type=int,
                        help="number of total steps to train for (num_batches*num_epochs)")
    parser.add_argument("--start-epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", default=256, type=int, help="total batch size")
    # Data/paths ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--data", default="tiered-imagenet-224",
                        help="id of the dataset to use: | ".join(DATASET_NAMES))
    parser.add_argument("--target_size", default=224, type=int,
                        help="Size of image input to the network (target resize after data augmentation)")
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="../../data_paths.yml")
    parser.add_argument("--data-path", default=None,
                        help="explicit location of the data folder, if None use config file.")
    parser.add_argument("--data_dir", default="../../data/", help="Folder containing the supplementary data")
    parser.add_argument("--output", default=None, help="path to the model folder")
    parser.add_argument("--expm_id", default="", type=str,
                        help="Name log folder as: out/<scriptname>/<date>_<expm_id>. If empty, expm_id=time")
    # Log/val -------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--log_freq", default=100, type=int, help="Log every log_freq batches")
    parser.add_argument("--val_freq", default=5, type=int,
                        help="Validate every val_freq epochs (except the first 10 and last 10)")
    # Execution -----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--workers", default=2, type=int, help="number of data loading workers")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")

    opts = parser.parse_args()

    # setup output folder
    opts.out_folder = opts.output if opts.output else get_expm_folder(__file__, "out", opts.expm_id)
    if not os.path.exists(opts.out_folder):
        print("Making experiment folder and subfolders under: ", opts.out_folder)
        os.makedirs(os.path.join(opts.out_folder, "json/train"))
        os.makedirs(os.path.join(opts.out_folder, "json/val"))
        os.makedirs(os.path.join(opts.out_folder, "model_snapshots"))


    # print options as dictionary and save to output
    PrettyPrinter(indent=4).pprint(vars(opts))
    with open(os.path.join(opts.out_folder, "opts.json"), "w") as fp:
        json.dump(vars(opts), fp)

    # setup data path from config file if needed
    if opts.data_path is None:
        opts.data_paths = load_config(opts.data_paths_config)
        opts.data_path = opts.data_paths[opts.data]

    # setup random number generation
    if opts.seed is not None:
        make_deterministic(opts.seed)

    gpus_per_node = torch.cuda.device_count()
    main_worker(gpus_per_node, opts)
