from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR

# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store
import numpy as np
import torch

# Hard-coded dataset, architecture, batch size, workers
ds = CIFAR('/home/zhuzby/data')
m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)

# Create a cox store for logging
out_store = cox.store.Store('coxx')

# Hard-coded base parameters
train_kwargs = {
    'out_dir': "train_out",
    'adv_train': 0,
    'lr': .01,
    'epochs': 300,
    'save-ckpt-iters': 10
}
train_args = Parameters(train_kwargs)

idx = np.arange(10)
order = np.load('../data/rnd_label_c10_5.npy')[idx].T
train_crit = torch.nn.BCELoss()
def custom_train_loss(logits, targ):
    if torch.cuda.is_available():
        targets = torch.from_numpy(order[targ.cpu().numpy()]).cuda()
    else:
        targets = torch.from_numpy(order[targ.numpy()])
    outputs = torch.sigmoid(logits.float())
    return train_crit(outputs.float(), targets.float())
train_args.custom_train_loss = custom_train_loss

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, CIFAR)
# train_args = defaults.check_and_fill_args(train_args,
                        # defaults.PGD_ARGS, CIFAR)

# Train a model
train.train_model(train_args, m, (train_loader, val_loader), store=out_store)
# from robustness.datasets import DATASETS
# from robustness.model_utils import make_and_restore_model
# from robustness.train import train_model
# from robustness.defaults import check_and_fill_args
# from robustness.tools import constants, helpers
# from robustness import defaults

# from cox import utils
# from cox import store

# import torch as ch
# from argparse import ArgumentParser
# import os

# parser = ArgumentParser()
# parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
# parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
# # parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
# # Note that we can add whatever extra arguments we want to the parser here
# train_args = parser.parse_args()


# # Fill whatever parameters are missing from the defaults
# train_args = defaults.check_and_fill_args(train_args,
#                         defaults.TRAINING_ARGS, CIFAR)


# # Load up the dataset
# data_path = os.path.expandvars(args.data)
# dataset = DATASETS[args.dataset](data_path)

# # Make the data loaders
# train_loader, val_loader = dataset.make_loaders(args.workers,
#               args.batch_size, data_aug=bool(args.data_aug))

# # Prefetches data to improve performance
# train_loader = helpers.DataPrefetcher(train_loader)
# val_loader = helpers.DataPrefetcher(val_loader)

# # Create the cox store, and save the arguments in a table
# store = store.Store(args.out_dir, args.exp_name)
# args_dict = args.as_dict() if isinstance(args, utils.Parameters) else vars(args)
# schema = store.schema_from_dict(args_dict)
# store.add_table('metadata', schema)
# store['metadata'].append_row(args_dict)

# model = train_model(args, model, (train_loader, val_loader), store=store)