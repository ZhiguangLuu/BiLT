import os
import torch
import requests
import torch.nn
import torch.cuda
import numpy as np
from tqdm import tqdm
from torchvision import models
from util.arch import wideresnet, custom_wideresnet, custom_resnet, HAF_resnet
from HAFrame.solve_HAF import distance_matrix_to_haf_cls_weights


def extract_resnet_state_dict(checkpoint):
    ckpt_dict = checkpoint["state_dict"]
    ret_dict = dict()
    for k in list(ckpt_dict.keys()):
        if k.startswith("module.encoder_q.") and not k.startswith("module.encoder_q.fc"):
            # remove prefix
            ret_dict[k[len("module.encoder_q."):]] = ckpt_dict[k]
    return ret_dict


def download_pass_pretrained_resnet50_weights(opts):
    url = "https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/moco_v2_800ep.pth.tar"
    file_path = os.path.join(opts.data_dir, "tiered-imagenet-224", "moco_v2_800ep.pth.tar")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise RuntimeError("ERROR, something went wrong while downloading the PASS pretrained weights of ResNet50")
    return


def load_pretrained_resnet50(opts):
    if opts.data in ["inaturalist19-224", "fgvc-aircraft"]:
        # load imagenet pretrained resnet50
        print("loading ImageNet pretrained resnet50 >>>>>>>>>>")
        return models.resnet50(pretrained=True)
    elif opts.data == "tiered-imagenet-224":
        # load PASS pretrained resnet50
        # https://github.com/yukimasano/PASS: PASS - MoCo-v2 - R50 - epoch-800
        ckpt_path = os.path.join(opts.data_dir, "tiered-imagenet-224", "moco_v2_800ep.pth.tar")
        if not os.path.exists(ckpt_path):
            download_pass_pretrained_resnet50_weights(opts)
        ckpt = torch.load(ckpt_path)
        encoder_state_dict = extract_resnet_state_dict(ckpt)
        model = models.resnet50(pretrained=False)
        model.load_state_dict(encoder_state_dict, strict=False)
        print("loading PASS pretrained resnet50 >>>>>>>>>>")
        return model
    else:
        raise ValueError(f"unrecognized opts.data {opts.data}")


def init_model_on_gpu(gpus_per_node, opts, distances=None):
    arch_dict = models.__dict__
    pretrained = False if not hasattr(opts, "pretrained") else opts.pretrained
    print("=> using model '{}', pretrained={}".format(opts.arch, pretrained))
    feature_dim = 512
    if opts.arch == "resnet18":
        feature_dim = 512
    elif opts.arch == "resnet50":
        feature_dim = 2048
    else:
        ValueError("Unknown architecture ", opts.arch)

    try:
        model = arch_dict[opts.arch](pretrained=pretrained)
    except:
        pass

    if opts.arch == "wide_resnet":
        model = wideresnet.WideResNet(num_classes=opts.num_classes)
        if opts.loss == "cross-entropy":
            model = custom_wideresnet.WideResNet(model, feature_size=512, num_classes=opts.num_classes)
        elif opts.loss == "flamingo-l5":
            model = custom_wideresnet.WideResNet_flamingo_l5(model, feature_size=512, num_classes=[100, 20, 8, 4, 2], gpu=opts.gpu)
        elif opts.loss == "hafeat-l5-cejsd-wtconst-dissim":
            model = custom_wideresnet.WideResNet_hafeat_l5_cejsd_wtconst_dissim(model, feature_size=512, num_classes=[100, 20, 8, 4, 2], gpu=opts.gpu)
        elif opts.loss == "ours_l5":
            model = custom_wideresnet.WideResNet_ours_l5_norm(model, feature_size=512, num_classes=[100, 20, 8, 4, 2], gpu=opts.gpu, alpha=opts.alpha, beta=opts.beta, epsilon=opts.epsilon, gamma=opts.gamma, weighting=opts.weighting)
    elif opts.arch == "haframe_wide_resnet":
        model = wideresnet.HAFrameWideResNet(num_classes=opts.num_classes)
        if opts.loss == "cross-entropy":
            model = custom_wideresnet.HAFrameWideResNet(model, num_classes=opts.num_classes, haf_cls_weights=None)
        elif opts.loss == "mixed-ce-gscsl":
            assert opts.haf_gamma > 0
            distance_matrix = np.array([[distances[c1, c2] for c1 in opts.class_str_labels] for c2 in opts.class_str_labels])
            haf_cls_weights, _, _ = distance_matrix_to_haf_cls_weights(distance_matrix, opts.class_str_labels, opts.num_classes, opts.haf_gamma)
            model = custom_wideresnet.HAFrameWideResNet(model, num_classes=opts.num_classes, haf_cls_weights=haf_cls_weights)
        elif opts.loss == "hafeat-l5-cejsd-wtconst-dissim":
            model = custom_wideresnet.HAFrameWideResNet_hafeat_l5_cejsd_wtconst_dissim(model, feature_size=opts.num_classes, num_classes=[100, 20, 8, 4, 2], gpu=opts.gpu)
        elif opts.loss == "flamingo-l5":
            model = custom_wideresnet.HAFrameWideResNet_flamingo_l5(model, feature_size=opts.num_classes, num_classes=[100, 20, 8, 4, 2], gpu=opts.gpu)
        elif opts.loss == "ours_l5":
            model = custom_wideresnet.HAFrameWideResNet_ours_l5_norm(model, feature_size=opts.num_classes, num_classes=[100, 20, 8, 4, 2], gpu=opts.gpu, alpha=opts.alpha, beta=opts.beta, epsilon=opts.epsilon, gamma=opts.gamma, weighting=opts.weighting)

    elif opts.arch == "custom_resnet50":
        # model = models.resnet50(pretrained=True)
        model = load_pretrained_resnet50(opts)
        if opts.loss == "cross-entropy":
            model = custom_resnet.ResNet50(model, feature_size=opts.hidden_units, num_classes=opts.num_classes)
        elif opts.loss == "hafeat-l3-cejsd-wtconst-dissim":
            model = custom_resnet.ResNet50_hafeat_l3_cejsd_wtconst_dissim(model, feature_size=opts.hidden_units, num_classes=[100, 70, 30], gpu=opts.gpu)
        elif opts.loss == "hafeat-l7-cejsd-wtconst-dissim":
            model = custom_resnet.ResNet50_hafeat_l7_cejsd_wtconst_dissim(model, feature_size=opts.hidden_units, num_classes=[1010, 72, 57, 34, 9, 4, 3], gpu=opts.gpu)
        elif opts.loss == "hafeat-l12-cejsd-wtconst-dissim":
            model = custom_resnet.ResNet50_hafeat_l12_cejsd_wtconst_dissim(model, feature_size=opts.hidden_units, num_classes=[608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2], gpu=opts.gpu)
        elif opts.loss == "flamingo-l3":
            model = custom_resnet.ResNet50_flamingo_l3(model, feature_size=opts.hidden_units, num_classes=[100, 70, 30], gpu=opts.gpu)
        elif opts.loss == "flamingo-l7":
            model = custom_resnet.ResNet50_flamingo_l7(model, feature_size=opts.hidden_units, num_classes=[1010, 72, 57, 34, 9, 4, 3], gpu=opts.gpu)
        elif opts.loss == "flamingo-l12":
            model = custom_resnet.ResNet50_flamingo_l12(model, feature_size=opts.hidden_units, num_classes=[608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2], gpu=opts.gpu)
        elif opts.loss == "ours_l3":
            model = custom_resnet.ResNet50_ours_l3_norm(model, feature_size=opts.hidden_units, num_classes=[100, 70, 30], gpu=opts.gpu, alpha=opts.alpha, beta=opts.beta, epsilon=opts.epsilon, gamma=opts.gamma, weighting=opts.weighting)
        elif opts.loss == "ours_l7":
            model = custom_resnet.ResNet50_ours_l7_norm(model, feature_size=opts.hidden_units, num_classes=[1010, 72, 57, 34, 9, 4, 3], gpu=opts.gpu, alpha=opts.alpha, beta=opts.beta, epsilon=opts.epsilon, gamma=opts.gamma, weighting=opts.weighting)
        elif opts.loss == "ours_l12":
            model = custom_resnet.ResNet50_ours_l12_norm(model, feature_size=opts.hidden_units, num_classes=[608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2], gpu=opts.gpu, alpha=opts.alpha, beta=opts.beta, epsilon=opts.epsilon, gamma=opts.gamma, weighting=opts.weighting)

    elif opts.arch == "haframe_resnet50":
        model = load_pretrained_resnet50(opts)
        if opts.loss == "cross-entropy":
            print(f"use {opts.pool} pooling in resnet")
            model = HAF_resnet.HAFrameResNet50(opts.pool, model, num_classes=opts.num_classes, haf_cls_weights=None)
        elif opts.loss == "mixed-ce-gscsl" or opts.loss == "fixed-ce":
            assert opts.haf_gamma > 0
            distance_matrix = np.array([[distances[c1, c2] for c1 in opts.class_str_labels] for c2 in opts.class_str_labels])
            haf_cls_weights, _, _ = distance_matrix_to_haf_cls_weights(distance_matrix, opts.class_str_labels, opts.num_classes, opts.haf_gamma)
            print(f"use {opts.pool} pooling in resnet")
            model = HAF_resnet.HAFrameResNet50(opts.pool, model, num_classes=opts.num_classes, haf_cls_weights=haf_cls_weights)
        elif opts.loss == "hafeat-l3-cejsd-wtconst-dissim":
            print(f"use {opts.pool} pooling in resnet")
            model = custom_resnet.HAFrameResNet50_hafeat_l3_cejsd_wtconst_dissim(opts.pool, model, feature_size=100, num_classes=[100, 70, 30], gpu=opts.gpu)
        elif opts.loss == "hafeat-l7-cejsd-wtconst-dissim":
            print(f"use {opts.pool} pooling in resnet")
            model = custom_resnet.HAFrameResNet50_hafeat_l7_cejsd_wtconst_dissim(opts.pool, model, feature_size=1010, num_classes=[1010, 72, 57, 34, 9, 4, 3], gpu=opts.gpu)
        elif opts.loss == "hafeat-l12-cejsd-wtconst-dissim":
            print(f"use {opts.pool} pooling in resnet")
            model = custom_resnet.HAFrameResnet50_hafeat_l12_cejsd_wtconst_dissim(opts.pool, model, feature_size=608, num_classes=[608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2], gpu=opts.gpu)
        elif opts.loss == "flamingo-l3":
            print(f"use {opts.pool} pooling in resnet")
            model = custom_resnet.HAFrameResNet50_flamingo_l3(opts.pool, model, feature_size=100, num_classes=[100, 70, 30], gpu=opts.gpu)
        elif opts.loss == "flamingo-l7":
            print(f"use {opts.pool} pooling in resnet")
            model = custom_resnet.HAFrameResNet50_flamingo_l7(opts.pool, model, feature_size=1010, num_classes=[1010, 72, 57, 34, 9, 4, 3], gpu=opts.gpu)
        elif opts.loss == "flamingo-l12":
            print(f"use {opts.pool} pooling in resnet")
            model = custom_resnet.HAFrameResNet50_flamingo_l12(opts.pool, model, feature_size=608, num_classes=[608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2], gpu=opts.gpu)
        elif opts.loss == "ours_l3_ot":
            print(f"use {opts.pool} pooling in resnet")
            model = custom_resnet.HAFrameResNet50_ours_l3(opts.pool, model, feature_size=100, num_classes=[100, 70, 30], gpu=opts.gpu, alpha=opts.alpha, beta=opts.beta, epsilon=opts.epsilon, gamma=opts.gamma, weighting=opts.weighting)
        elif opts.loss == "ours_l7_ot":
            print(f"use {opts.pool} pooling in resnet")
            model = custom_resnet.HAFrameResNet50_ours_l7(opts.pool, model, feature_size=1010, num_classes=[1010, 72, 57, 34, 9, 4, 3], gpu=opts.gpu, alpha=opts.alpha, beta=opts.beta, epsilon=opts.epsilon, gamma=opts.gamma, weighting=opts.weighting)
        elif opts.loss == "ours_l12_ot":
            print(f"use {opts.pool} pooling in resnet")
            model = custom_resnet.HAFrameResNet50_ours_l12(opts.pool, model, feature_size=608, num_classes=[608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2], gpu=opts.gpu, alpha=opts.alpha, beta=opts.beta, epsilon=opts.epsilon, gamma=opts.gamma, weighting=opts.weighting)

    else:
        model.fc = torch.nn.Sequential(torch.nn.Dropout(opts.dropout), torch.nn.Linear(in_features=feature_dim, out_features=opts.num_classes, bias=True) )

    if opts.gpu is not None:
        torch.cuda.set_device(opts.gpu)
        model = model.cuda(opts.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    return model
