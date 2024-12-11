# import json
# import os
# from datetime import datetime
# import wandb

# # topK_to_consider = (1, 5, 10, 20, 100)
# topK_to_consider = (1, )

# accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
# dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
# dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
# dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
# hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
# hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]


# def init(project, entity=None, config=None, run_name=None):
#     wandb.init(project=project, config=config)
#     if run_name:
#         wandb.run.name = run_name + "(" + datetime.today().strftime('%Y-%m-%d_%H%M') + ")"


# def log(out_folder, summary, epoch, is_training=False, is_validation=False, is_testing=False, save_log=False):
#     # name for the json file s
#     json_name = "epoch.%04d.json" % epoch

#     if is_validation:
#         log = {}
#         for key in summary:
#             log["validation/" + key] = summary[key]

#         # log for validation
#         for i in range(len(topK_to_consider)):
#             wandb.log(log, step=epoch)

#         # dump results
#         if save_log:
#             with open(os.path.join(out_folder, "json/val", json_name), "w") as fp:
#                 json.dump(summary, fp)
#     elif is_training:
#         # log for training
#         log = {}
#         for key in summary:
#             log["train/" + key] = summary[key]
#         for i in range(len(topK_to_consider)):
#             wandb.log(log, step=epoch)

#         # dump results
#         if save_log:
#             with open(os.path.join(out_folder, "json/train", json_name), "w") as fp:
#                 json.dump(summary, fp)
#     elif is_testing:
#         with open(os.path.join(out_folder, "json/test", json_name), "w") as fp:
#             json.dump(summary, fp)


# def _print(log, file_name):
#     print(log)
#     # with open(file_name, "a") as log_file:
#     #     log_file.write(log+"\n")

import json
import os
from datetime import datetime
# import wandb
import logging

# topK_to_consider = (1, 5, 10, 20, 100)
topK_to_consider = (1, )

accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]

def logger_init(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, name+'.log'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger
