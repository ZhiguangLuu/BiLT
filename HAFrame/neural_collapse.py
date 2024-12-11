import os
import torch
import numpy as np
import pickle

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# HAFeature
from MBM.better_mistakes.util.config import load_config
from MBM.better_mistakes.data.transforms import val_transforms
from MBM.better_mistakes.model.init import init_model_on_gpu
from MBM.better_mistakes.trees import load_distances
from util.data_loader import ToTensor, CIFAR100_labeled, train_val_split


def cosine_similarity(x, y):
    return np.sum(x * y)/(np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))


def pair_wise_cosine_similarity(vectors):
    """
    input:
        vectors: centered class feature means or classifier weights
    return:
        cos_similarities: a list of pair-wise cosine similarities
    """
    num_classes, num_features = vectors.shape
    cos_similarities = []

    for i in range(num_classes-1):
        for j in range(i+1, num_classes):
            cos_similarities.append(cosine_similarity(vectors[i], vectors[j]))

    cos_similarities = np.array(cos_similarities)

    return cos_similarities


def stats_between_actual_cosine_sim_and_ideal_cosine_sim(opts, vectors, cls_weights):
    num_classes = vectors.shape[0]
    vect_similarities = pair_wise_cosine_similarity(vectors)
    if opts.frame == 'ETF':
        shift_sim_mean = np.mean(np.abs(vect_similarities + 1.0/(num_classes -1)))
        sim_std = np.std(vect_similarities)
    elif opts.frame == 'HAF':
        # cls_weights are fixed to pre-computed ideal HAFrame
        classifier_similarities = pair_wise_cosine_similarity(cls_weights)
        shift_sim_mean = np.mean(np.abs(vect_similarities - classifier_similarities))
        sim_std = np.std(vect_similarities - classifier_similarities)
    else:
        raise NotImplementedError(f"frame '{opts.frame}', not recognized")
    return shift_sim_mean, sim_std


def cls_weights_and_feature_collapse(cls_weights, class_features):
    """
        self duality
    """

    cls_weights_normalized = cls_weights / np.linalg.norm(cls_weights, ord='fro')
    class_features_normalized = class_features / np.linalg.norm(class_features, ord='fro')
    diff = np.linalg.norm(cls_weights_normalized - class_features_normalized, ord='fro')
    return diff**2


def extract_classifier_weights(model):
    print("\textract class weights")
    cls_weights = None
    for name, param in model.named_parameters():
        if name == "classifier_3.weight" : # 'fc.weight'
            cls_weights = param.cpu().data.numpy()

    if cls_weights is None:
        raise ValueError("'classifier_3.weight' not found")

    return cls_weights


def extract_class_feature_means(model, data_loader, feature_option, device):
    """
    input:
        model: pretrained model
        data_loader: training data loader
        device: device of model, 'cuda', 'cpu', etc
    return:
        class_feature_mean: mean of penultimate features for each class, shape (num_classes, num_features)
        global_feature_mean: mean of all penultimate features, shape (1, num_features)
    """
    model.eval()
    num_classes = model.classifier_3.out_features  # model.fc.out_features
    features = []
    labels = []
    print("\textract class feature means")
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(data_loader):
            input_var = inputs.to(device)
            if feature_option == 'penultimate':
                embeddings = model.penultimate_feature(input_var)
            elif feature_option == 'backbone':
                embeddings = model.backbone_feature(input_var)
            else:
                raise ValueError(f"unrecognized feature_option '{feature_option}'")

            # collect penultimate features and their labels
            features.append(embeddings.cpu().data.numpy())
            labels.append(targets.data.numpy())

    # convert to ndarray
    features_np = np.concatenate(features, 0)
    labels_np = np.concatenate(labels, 0)

    # feature global mean
    global_features_mean = np.mean(features_np, 0, keepdims=True)

    class_features_mean = []

    # class features mean
    for k in range(num_classes):
        idx = labels_np == k
        class_features = features_np[idx, :]
        class_features_mean.append(np.mean(class_features, 0, keepdims=True))

    class_features_mean = np.concatenate(class_features_mean, 0)
    return class_features_mean, global_features_mean


def extract_class_feature_means_cumulative(model, data_loader, feature_option, device):
    """
    input:
        model: pretrained model
        data_loader: training data loader
        device: device of model, 'cuda', 'cpu', etc
    return:
        class_feature_mean: mean of penultimate features for each class, shape (num_classes, num_features)
        global_feature_mean: mean of all penultimate features, shape (1, num_features)
    """
    model.eval()
    num_features = model.classifier_3.in_features  # model.fc.in_features
    num_classes = model.classifier_3.out_features  # model.fc.out_features

    global_feature_sum = torch.zeros([1, num_features]).to(device)
    class_feature_sums = torch.zeros([num_classes, num_features]).to(device)
    print("\textract class feature means in a cumulative fashion")
    example_cnt = 0
    class_example_cnt = np.zeros([num_classes, 1])
    with torch.no_grad():
        for k, (inputs, targets) in enumerate(data_loader):
            example_cnt += inputs.shape[0]
            inputs_var = inputs.to(device)
            if feature_option == 'penultimate':
                embeddings = model.penultimate_feature(inputs_var)
            elif feature_option == 'backbone':
                embeddings = model.backbone_feature(inputs_var)
            else:
                raise ValueError(f"unrecognized feature_option '{feature_option}'")

            # update global feature mean
            global_feature_sum += embeddings.sum(dim=0, keepdim=True)

            # update class feature means
            for i in range(num_classes):
                cnt = torch.sum(targets == i)
                # if cnt == 0, class_features == tensor([], size=(0, num_features))
                class_features = embeddings[targets == i, :]
                class_example_cnt[i][0] += cnt
                class_feature_sums[i, :] += class_features.sum(dim=0, keepdim=False)

    global_feature_mean = global_feature_sum.cpu().data.numpy() / example_cnt
    class_feature_means = class_feature_sums.cpu().data.numpy() / class_example_cnt

    return class_feature_means, global_feature_mean


def collect_neural_collapse_stats(record_dir, model, data_loader, epochs, opts):

    # stats of classifier weights
    cls_cos_mean = np.zeros([len(epochs), ])
    cls_cos_std = np.zeros([len(epochs), ])

    # stats of [centered] features
    cfeat_cos_mean = np.zeros([len(epochs), ])
    cfeat_cos_std = np.zeros([len(epochs), ])
    
    # stats of [raw] features
    rfeat_cos_mean = np.zeros([len(epochs), ])
    rfeat_cos_std = np.zeros([len(epochs), ])

    # self duality
    cls_cfeat_duality = np.zeros([len(epochs), ])
    cls_rfeat_duality = np.zeros([len(epochs), ])

    for i, e in enumerate(epochs):
        print(f"start epoch: {i+1}/{len(epochs)}")

        # load model
        model_path = os.path.join(record_dir, "checkpoints", f"epoch_{e}_checkpoint.pth.tar")
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt["state_dict"])

        # extract classifier weights, compute statistics
        cls_weights = extract_classifier_weights(model)
        cls_cos_mean[i], cls_cos_std[i] = \
            stats_between_actual_cosine_sim_and_ideal_cosine_sim(
                opts,
                cls_weights,
                cls_weights
            )

        # extract class features, compute statistics
        if opts.data == "cifar-100":
            # raw class feature mean
            cls_rfeature_mean, global_feature_mean = \
                extract_class_feature_means(
                    model,
                    data_loader,
                    "penultimate",
                    opts.gpu
                )
        else:
            # raw class feature mean
            cls_rfeature_mean, global_feature_mean = \
                extract_class_feature_means_cumulative(
                    model,
                    data_loader,
                    "penultimate",
                    opts.gpu
                )

        # centered class feature means
        cls_cfeature_mean = cls_rfeature_mean - global_feature_mean

        # stats of [centered] features
        cfeat_cos_mean[i], cfeat_cos_std[i] = \
            stats_between_actual_cosine_sim_and_ideal_cosine_sim(
                opts,
                cls_cfeature_mean,
                cls_weights
            )

        # stats of [raw] features
        rfeat_cos_mean[i], rfeat_cos_std[i] = \
            stats_between_actual_cosine_sim_and_ideal_cosine_sim(
                opts,
                cls_rfeature_mean,
                cls_weights
            )

        # self duality
        cls_cfeat_duality[i] = cls_weights_and_feature_collapse(cls_weights, cls_cfeature_mean)
        cls_rfeat_duality[i] = cls_weights_and_feature_collapse(cls_weights, cls_rfeature_mean)

    ret_dict = {
        'cls_cos_mean': cls_cos_mean,
        'cls_cos_std': cls_cos_std,
        'cfeat_cos_mean': cfeat_cos_mean,
        'cfeat_cos_std':cfeat_cos_std,
        'rfeat_cos_mean': rfeat_cos_mean,
        'rfeat_cos_std':rfeat_cos_std,
        'cls_cfeat_duality':cls_cfeat_duality,
        'cls_rfeat_duality':cls_rfeat_duality,
    }

    return ret_dict


def viz_neural_collapse(opts):
    gpus_per_node = torch.cuda.device_count()
    assert opts.out_folder

    # run_name = f"{opts.loss}-{opts.arch}"

    # setup data path from config file if needed
    if opts.data_path is None:
        opts.data_paths = load_config(opts.data_paths_config)
        opts.data_path = opts.data_paths[opts.data]

    # evaluation transforms + evaluation/target dataset
    if opts.data == "cifar-100":
        eval_transform = transforms.Compose([ToTensor()])
        base_dataset = torchvision.datasets.CIFAR100(opts.data_path, train=True, download=True)
        train_idxs, val_idxs = train_val_split(base_dataset.targets, len(base_dataset.classes))
        if opts.partition == "train":
            example_idxs = train_idxs
        else:
            example_idxs = val_idxs
        target_dataset = CIFAR100_labeled(
                                opts.data_path,
                                example_idxs,
                                train=True,  # either train, or val all came from original train set of CIFAR100
                                transform=eval_transform
                            )
    else:
        eval_transform = val_transforms(opts.data, normalize=True, resize=opts.target_size)
        if opts.partition == "train":
            target_data_dir = os.path.join(opts.data_path, "train_images")
        else:
            target_data_dir = os.path.join(opts.data_path, "val_images")
        target_dataset = datasets.ImageFolder(
                                    target_data_dir,
                                    eval_transform
                                )

    # data loader
    data_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.workers,
        pin_memory=False,  # save shared memory
        drop_last=False
    )

    if opts.data == "cifar-100":
        classes = target_dataset.class_to_idx
        classes = ["L5-" + str(classes[i]) for i in classes]
    else:
        classes = target_dataset.classes

    num_classes = len(target_dataset.classes)
    opts.num_classes = num_classes

    # load distances
    if opts.data == "fgvc-aircraft":
        distances = load_distances(opts.data, 'original', opts.data_dir)
    else:
        distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)

    # carry class string labels with opts
    opts.class_str_labels = classes

    # setup model
    model = init_model_on_gpu(gpus_per_node, opts, distances)

    # plot neural collapse visualization
    epochs = np.arange(opts.ckpt_freq, opts.epochs, opts.ckpt_freq)

    # collect mean and std of differences between actual and ideal cosine similarities
    ret_stats = collect_neural_collapse_stats(
        opts.out_folder,
        model,
        data_loader,
        epochs,
        opts
    )

    # save viz results
    nc_result_path = os.path.join(opts.out_folder, f"{opts.partition}_penultimate_feature_neural_collapse.pkl")
    with open(nc_result_path, "wb") as f:
        pickle.dump(ret_stats, f)
    print(f"saving nc results at {nc_result_path}")

    return

# EOF
