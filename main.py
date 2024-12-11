import os
import json
import argparse

from distutils.util import strtobool as boolean
from pprint import PrettyPrinter

import torch.utils.data.distributed
import torchvision.models as models

from MBM.better_mistakes.util.rand import make_deterministic
from MBM.better_mistakes.util.folders import get_expm_folder
from MBM.better_mistakes.util.config import load_config

from util import logger
from HAFrame import neural_collapse
from MBM.scripts import start_training, start_testing, crm_eval

TASKS = ["training", "validating", "testing", "crm", "neural-collapse"]
CUSTOM_MODELS = ["custom_resnet50", "haframe_resnet50", "wide_resnet", "haframe_wide_resnet"]
MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
MODEL_NAMES.extend(CUSTOM_MODELS)

# l5 refers to loss of level-5 for CIFAR-100,
# l7 refers to loss of level-7 for iNaturalist-19,
# l12 refers to loss of level-12 for tiered-imageneget-224.
LOSS_NAMES = [
    "cross-entropy", # cross-entropy baseline
    "flamingo-l3", "flamingo-l5", "flamingo-l7", "flamingo-l12", # Flamingo
    "hafeat-l3-cejsd-wtconst-dissim", "hafeat-l5-cejsd-wtconst-dissim", # HAFeature with all three losses
    "hafeat-l7-cejsd-wtconst-dissim", "hafeat-l12-cejsd-wtconst-dissim",
    "mixed-ce-gscsl", # HAFrame loss
    "ours_l3", "ours_l5", "ours_l7", "ours_l12"
]
OPTIMIZER_NAMES = ["adagrad", "adam", "adam_amsgrad", "rmsprop", "SGD", "custom_sgd", "step_sgd"]
DATASET_NAMES = [ "tiered-imagenet-224", "inaturalist19-224", "cifar-100", "fgvc-aircraft" ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="training", choices=TASKS, help="name of the task | ".join(TASKS))
    parser.add_argument("--arch", default="custom_resnet50", choices=MODEL_NAMES, help="model architecture: | ".join(MODEL_NAMES))
    parser.add_argument("--hidden_units", type=int, default=600, help="number of hidden units in the transformation layer of custom_resnet")
    parser.add_argument("--loss", default="cross-entropy", choices=LOSS_NAMES, help="loss type: | ".join(LOSS_NAMES))
    parser.add_argument("--optimizer", default="adam_amsgrad", choices=OPTIMIZER_NAMES, help="optimizer type: | ".join(OPTIMIZER_NAMES))
    parser.add_argument("--lr", default=1e-5, type=float, help="initial learning rate of optimizer")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay of optimizer")
    parser.add_argument("--pretrained", type=boolean, default=True, help="start from ilsvrc12/imagenet model weights")
    parser.add_argument("--pretrained_folder", type=str, default=None, help="folder or file from which to load the network weights")
    parser.add_argument("--dropout", default=0.0, type=float, help="Prob of dropout for network FC layer")
    parser.add_argument("--data_augmentation", type=boolean, default=True, help="Train with basic data augmentation")
    parser.add_argument("--epochs", default=None, type=int, help="number of epochs")
    parser.add_argument("--num_training_steps", default=200000, type=int, help="number of total steps to train for (num_batches*num_epochs)")
    parser.add_argument("--start-epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", default=256, type=int, help="total batch size")
    # Data/paths ----------------------------------------------------------------------------------------------------- #
    parser.add_argument("--data", default="inaturalist19-224", help="id of the dataset to use: | ".join(DATASET_NAMES))
    parser.add_argument("--target_size", default=224, type=int, help="Size of image input to the network (target resize after data augmentation)")
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="data_paths.yml")
    parser.add_argument("--data-path", default=None, help="explicit location of the data folder, if None use config file.")
    parser.add_argument("--data_dir", default="data/", help="Folder containing the supplementary data")
    parser.add_argument("--output", default="ckpt/", help="path to the model folder")
    parser.add_argument("--expm_id", default="", type=str, help="Name log folder as: out/<scriptname>/<date>_<expm_id>. If empty, expm_id=time")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="path to the best checkpoint file")
    # Log/val -------------------------------------------------------------------------------------------------------- #
    parser.add_argument("--log_freq", default=100, type=int, help="Log every log_freq batches")
    parser.add_argument("--val_freq", default=1, type=int, help="Validate every val_freq epochs (except the first 10 and last 10)")
    # Execution ------------------------------------------------------------------------------------------------------ #
    parser.add_argument("--workers", default=32, type=int, help="number of data loading workers")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    # HAFrame -------------------------------------------------------------------------------------------------------- #
    parser.add_argument("--loss-schedule", type=str, default=None)
    parser.add_argument("--loss-schedule-period", type=str, default=None)
    parser.add_argument("--haf-gamma", type=float, default=None)
    parser.add_argument("--larger-backbone", action="store_true", help="set model.features_1's lr to be the same as model.features_2 (backbone)")
    parser.add_argument("--raise-lower-bound", action="store_true", help="raise cosine annealling lr schedule lower bound to 1e-4")
    parser.add_argument("--ckpt-freq", type=int, default=0, help="if ckpt-freq != 0, save checkpoint periodically")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--pool", type=str, default="max")
    # Neural Collapse ------------------------------------------------------------------------------------------------ #
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--frame", type=str, default=None) # has to explicitly specify
    # ot args -------------------------------------------------------------------------------------------------------- #
    parser.add_argument("--weighting", type=str, default="exp", help="Decay parameter for hierarchy.")
    parser.add_argument("--alpha", type=float, default=0.1, help="alpha")
    parser.add_argument("--beta", type=float, default=1, help="beta")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon")
    parser.add_argument("--gamma", type=float, default=1, help="gamma")

    opts = parser.parse_args()

    # setting the path of level-5 distances and pickle file.
    if opts.data == "cifar-100":
        opts.data_dir = os.path.join(opts.data_dir, "cifar-l5/original/")

    # setup output folder
    opts.out_folder = opts.output if opts.output else get_expm_folder(__file__, "out", opts.expm_id)
    if opts.loss.startswith("ours_"):
        opts.out_folder = opts.out_folder + f'_{opts.arch}_seed{opts.seed}_lr{opts.lr}_{opts.optimizer}_{opts.weighting}_alp{opts.alpha}_bet{opts.beta}_epc{opts.epochs}_ep{opts.epsilon}_gam{opts.gamma}'
    if not os.path.exists(opts.out_folder):
        print("Making experiment folder and subfolders under: ", opts.out_folder)
        os.makedirs(os.path.join(opts.out_folder))

    # if periodic checkpointing is turned on
    if opts.ckpt_freq != 0:
        if not os.path.exists(os.path.join(opts.out_folder, "checkpoints")):
            os.makedirs(os.path.join(opts.out_folder, "checkpoints"))
        else:
            print(f"checkpoints dir already exists")

    # print options as dictionary and save to output
    PrettyPrinter(indent=4).pprint(vars(opts))
    if opts.start == "training":
        # Create opts.json file
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

    if opts.start == "training":
        # Setup wandb logging parameters
        if opts.data == "cifar-100":
            project = "cifar-100"
        elif opts.data == "inaturalist19-224":
            project = "inaturalist19-224"
        elif opts.data == "tiered-imagenet-224":
            project = "tiered-imagenet-224"
        elif opts.data == "fgvc-aircraft":
            project = "fgvc-aircraft"
        else:
            raise ValueError(f"opts.data {opts.data} unknown")

        entity = "hierarchy-aware-classification"
        if opts.loss == "cross-entropy":
            run_name = f"cross-entropy-{opts.arch}"
        elif opts.loss.startswith("ours_"):
            run_name = opts.loss
        else:
            run_name = f"{opts.loss}-{opts.arch}"

        if opts.tag:  # additional tag
            project = f"{project}-{opts.tag}"

        if opts.seed is not None:
            run_name = f"{run_name}-seed_{opts.seed}"

        logger_name = entity + "-" + project + "-" + run_name 
        opts.logger_train = logger.logger_init(name=logger_name + '-train', save_path=opts.out_folder, level='INFO')
        opts.logger_val = logger.logger_init(name=logger_name + '-val', save_path=opts.out_folder, level='INFO')

        # Start training
        start_training.main_worker(gpus_per_node, opts)
    elif opts.start == "validating":  # use validation set for hyper-parameter selections
        logger_name = 'MBM-Validation-Results'
        opts.logger_val = logger.logger_init(name=logger_name + '-val', save_path=opts.out_folder, level='INFO')
        # reuse start_testing, but with val_loader
        start_testing.main(opts)
    elif opts.start == "testing":     # use test set for final evaluation
        logger_name = 'MBM-Test-Results'
        opts.logger_val = logger.logger_init(name=logger_name + '-test', save_path=opts.out_folder, level='INFO')
        start_testing.main(opts)
    elif opts.start == "neural-collapse":
        assert opts.frame is not None
        assert opts.partition is not None
        logger_name = 'MBM-Test-Results'
        opts.logger_val = logger.logger_init(name=logger_name + '-neural_collapse', save_path=opts.out_folder, level='INFO')
        neural_collapse.viz_neural_collapse(opts)
    else: # crm
        logger_name = 'MBM-CRM-Test-Results'
        opts.logger_val = logger.logger_init(name=logger_name + '-crm_test', save_path=opts.out_folder, level='INFO')
        crm_eval.main(opts)
