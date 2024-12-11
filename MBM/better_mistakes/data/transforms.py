"""
transformation for ImageFolder datasets: fgvc-aircraft, inaturalist19, tiered-imagenet
CIFAR-100 is using the dataset provided by torchvision
"""

from torchvision import transforms

mean_ilsvrc12 = [0.485, 0.456, 0.406]
std_ilsvrc12 = [0.229, 0.224, 0.225]

mean_inat19 = [0.454, 0.474, 0.367]
std_inat19 = [0.237, 0.230, 0.249]

mean_fgvc = [0.4880712, 0.5191783, 0.54383063]
std_fgvc = [0.21482512, 0.20683703, 0.23937796]


normalize_tfs_ilsvrc12 = transforms.Normalize(mean=mean_ilsvrc12, std=std_ilsvrc12)
normalize_tfs_inat19 = transforms.Normalize(mean=mean_inat19, std=std_inat19)
normalize_fgvc_aircraft = transforms.Normalize(mean=mean_fgvc, std=std_fgvc)

normalize_tfs_dict = {
    "tiered-imagenet-224": normalize_tfs_ilsvrc12,
    "inaturalist19-224": normalize_tfs_inat19,
    "fgvc-aircraft": normalize_fgvc_aircraft,
}


def train_transforms(img_resolution, dataset, augment=True, normalize=True):
    if dataset == "fgvc-aircraft":
        return transforms.Compose([
            transforms.Resize((img_resolution, img_resolution)),  # to deal with iNat2019 data
            transforms.RandomCrop(img_resolution, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_tfs_dict[dataset],
        ])
    if augment and normalize:
        return transforms.Compose(
            [
                # extract random crops and resize to img_resolution
                transforms.RandomResizedCrop(img_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_tfs_dict[dataset],
            ]
        )
    elif not augment and normalize:
        return transforms.Compose([transforms.ToTensor(), normalize_tfs_dict[dataset]])
    elif augment and not normalize:
        return transforms.Compose([transforms.RandomResizedCrop(img_resolution),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])
    else:
        return transforms.Compose([transforms.ToTensor()])


def val_transforms(dataset, normalize=True, resize=None, crop=None):
    trsfs = []

    if resize:
        trsfs.append(transforms.Resize((resize, resize)))

    if crop:
        trsfs.append(transforms.CenterCrop(crop))

    if normalize:
        trsfs.extend([transforms.ToTensor(), normalize_tfs_dict[dataset]])
    else:
        trsfs.append([*transforms.ToTensor()])

    return transforms.Compose(trsfs)
