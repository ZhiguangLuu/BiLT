import os
import json
import torch
import heapq
import numpy as np
from scipy.special import softmax

from util import data_loader, logger
from MBM.better_mistakes.trees import load_distances
from MBM.better_mistakes.model.init import init_model_on_gpu


def get_dataset_stats(opts):

    if opts.data == 'cifar-100':
        train_mean = [0.5071, 0.4867, 0.4408]
        train_std = [0.2675, 0.2565, 0.2761]
        input_size = 32
        num_classes = 100
    elif opts.data == 'inaturalist19-224':
        train_mean = [0.454, 0.474, 0.367]
        train_std = [0.237, 0.230, 0.249]
        input_size = 224
        num_classes = 1010
    elif opts.data == 'tiered-imagenet-224':
        train_mean = [0.485, 0.456, 0.406]
        train_std = [0.229, 0.224, 0.225]
        input_size = 224
        num_classes = 608
    elif opts.data == 'fgvc_aircraft':
        train_mean = [0.4880712, 0.5191783, 0.54383063]
        train_std = [0.21482512, 0.20683703, 0.23937796]
        input_size = 224
        num_classes = 100
    else:
        raise ValueError(f"mean and std for dataset {opts.dataset_name} unknown")

    return train_mean, train_std, input_size, num_classes


def logit_extraction(data_loader, model, opts, verbose=True):

    model.eval()
    num_batch = len(data_loader)
    # model.to(opts.gpu)
    # print(f"opts.gpu: {opts.gpu}")

    logit = []
    gt = []

    with torch.no_grad():
        for i, (input_var, target) in enumerate(data_loader):
            gt.append(target.data.numpy())
            target_var = target.type(torch.LongTensor).cuda(opts.gpu)
            input_var = input_var.to(opts.gpu)

            if opts.arch in ["custom_resnet50", "wide_resnet"]:
                if opts.loss in ["hafeat-l12-cejsd-wtconst-dissim",
                                 "hafeat-l7-cejsd-wtconst-dissim",
                                 "hafeat-l3-cejsd-wtconst-dissim"]:
                    logit_var = model(input_var, target_var, True)
                else:
                    logit_var = model(input_var, target_var)

            elif opts.arch == ["wide_resnet", "haframe_wide_resnet"]:
                logit_var = model(input_var, target_var)

            elif opts.arch == "haframe_resnet50":
                if opts.loss in ["hafeat-l12-cejsd-wtconst-dissim",
                                 "hafeat-l7-cejsd-wtconst-dissim",
                                 "hafeat-l3-cejsd-wtconst-dissim"]:
                    logit_var = model(input_var, target_var, True)
                else:
                    logit_var = model(input_var, target_var)

            else:
                logit_var = model(input_var)

            logit.append(logit_var.cpu().data.numpy())

            if verbose:
                print(f"{i+1}/{num_batch} completed")

    # convert to ndarray
    num_block = len(logit)
    logit_arr = logit[0]
    gt_arr = gt[0]
    for i in range(1, num_block):
        logit_arr = np.vstack([logit_arr, logit[i]])
        gt_arr = np.concatenate([gt_arr, gt[i]], axis=0)

    return logit_arr, gt_arr


# conditional risk minimization
def apply_crm_to_softmax(output, distances, classes):
    """
        Re-Rank all predictions in the dataset using CRM
        Args:
            output: softmax output from base CNN model
            distances: hierarchical distances between terminal/leaf classes
            classes: a list of class labels for all leaf classes
        """

    num_classes = len(classes)
    C = [[0 for i in range(num_classes)] for j in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            C[i][j] = distances[(classes[i], classes[j])]
    final = np.dot(output, C)

    return -1 * final


def get_topk_dist(prediction, target, distances, classes, k=1):
    """
    compute hierarchical distance@k
    Args:
        prediction: softmax/logits/negative conditional risk vector of shape (n_classes,)
        target: scalar
        distances: distance dictionary {(node_a,node_b):dist(a,b)}
        classes: class names, e.g., iNat2019 classes nat0000, nat0001, etc
        k: scalar, specify the topk predictions to compute hierarchical dist@k
    Returns:
        scores: a list of k hierarchical distances for input top k predictions
    """

    # return k largest element in prediction
    ind = heapq.nlargest(k, range(len(prediction)), prediction.take)
    scores = []

    for i in ind:
        scores.append(distances[(classes[i],classes[target])])

    return scores


# evaluation
def better_mistakes_metrics(output, target, distances, classes):
    top1 = []
    mistake = []
    hie_avg_1 = []
    hie_avg_5 = []
    if len(classes) >= 20:
        hie_avg_20 = []
    else:
        hie_avg_20 = None

    for i, op in enumerate(output):
        if op.argmax() == target[i]:
            top1.append(1)
        else:
            top1.append(0)
            mistake.append(distances[(classes[op.argmax()], classes[target[i]])])

        hie_avg_1.append(get_topk_dist(op, target[i], distances, classes, 1))
        hie_avg_5.append(get_topk_dist(op, target[i], distances, classes, 5))
        if len(classes) >= 20:
            hie_avg_20.append(get_topk_dist(op, target[i], distances, classes, 20))

    # compute metrics
    top1 = np.array(top1).mean()
    top1 = top1 * 100  # convert to %
    mistake_mean = np.array(mistake).mean()
    mistake_std = np.array(mistake).std()
    hie_avg_1 = np.array(hie_avg_1).mean()
    hie_avg_5 = np.array(hie_avg_5).mean()
    if len(classes) >= 20:
        hie_avg_20 = np.array(hie_avg_20).mean()

    if len(classes) >= 20:
        res_str = (f"top1 accuracy {top1:.3f}%\n"
                   f"mistake severity mean: {mistake_mean:.3f}\n"
                   f"mistake severity std: {mistake_std:.3f}\n"
                   f"average hierarchical distance @1: {hie_avg_1:.3f}\n"
                   f"average hierarchical distance @5: {hie_avg_5:.3f}\n"
                   f"average hierarchical distance @20: {hie_avg_20:.3f}\n"
                   )
    else:
        res_str = (f"top1 accuracy {top1:.3f}%\n"
                   f"mistake severity mean: {mistake_mean:.3f}\n"
                   f"mistake severity std: {mistake_std:.3f}\n"
                   f"average hierarchical distance @1: {hie_avg_1:.3f}\n"
                   f"average hierarchical distance @5: {hie_avg_5:.3f}\n"
                   )

    result = {
        'top1': top1,
        'mistake': mistake_mean,
        'mistake_std': mistake_std,
        'hie_avg_1': hie_avg_1,
        'hie_avg_5': hie_avg_5,
        'hie_avg_20': hie_avg_20,
        'res_str': res_str,
    }

    return result, res_str


def main(opts):
    gpus_per_node = torch.cuda.device_count()
    assert opts.out_folder

    # Setup data loaders
    test_dataset, test_loader = data_loader.test_data_loader(opts, False)

    # Load distance -------------------------------------------------------------------------------------------------- #
    if opts.data == "fgvc-aircraft":
        distances = load_distances(opts.data, 'original', opts.data_dir)
    else:
        distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)

    # get class names
    if opts.data == "cifar-100":
        classes = test_dataset.class_to_idx
        classes = ["L5-" + str(classes[i]) for i in classes]
    else:
        classes = test_dataset.classes

    opts.num_classes = len(classes)

    # carry class string labels with opts, used to init HAFrame model
    opts.class_str_labels = classes

    # setup model
    model = init_model_on_gpu(gpus_per_node, opts, distances)

    if opts.checkpoint_path is None:
        checkpoint_id = "best.checkpoint.pth.tar"
        checkpoint_path = os.path.join(opts.out_folder, checkpoint_id)
    else:
        checkpoint_path = opts.checkpoint_path

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        logger._print("=> loaded checkpoint '{}'".format(checkpoint_path), os.path.join(opts.out_folder, "logs.txt"))
    else:
        logger._print("=> no checkpoint found at '{}'".format(checkpoint_path), os.path.join(opts.out_folder, "logs.txt"))
        raise RuntimeError

    # turn to eval mode
    model.eval()

    # get raw logits and corresponding ground truths
    logit, gt = logit_extraction(test_loader, model, opts, True)
    pred = np.argmax(logit, axis=1)
    acc = np.sum(pred == gt) / len(gt)
    print(f"raw logits and gt acc: {acc:.4f}")

    # get softmax
    sm = softmax(logit, axis=1)

    eval_res_base, eval_str_base = better_mistakes_metrics(
        sm,
        gt,
        distances,
        classes
    )

    print(f"base predictions:\n{eval_str_base}")

    # get negative of the crm risks
    negative_crm_risk = apply_crm_to_softmax(sm, distances, classes)

    eval_res_crm, eval_str_crm = better_mistakes_metrics(
        negative_crm_risk,
        gt,
        distances,
        classes
    )

    print(f"crm predictions:\n{eval_str_crm}")

    summaries = dict()

    for k in eval_res_base.keys():
        val = eval_res_base[k]
        base_key = f"base_{k}"
        if base_key not in summaries:
            summaries[base_key] = []
        summaries[base_key].append(val)

    for k in eval_res_crm.keys():
        val = eval_res_crm[k]
        crm_key = f"crm_{k}"
        if crm_key not in summaries:
            summaries[crm_key] = []
        summaries[crm_key].append(val)

    with open(os.path.join(opts.out_folder, "base_n_crm_test_summary.json"), "w") as fp:
        json.dump(summaries, fp, indent=4)
