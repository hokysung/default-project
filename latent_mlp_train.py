import os
import pprint
import argparse

import torch
import torch.optim as optim

import util
from model import *
from latent_mlp_trainer import Trainer

import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    r"""
    Parses command line arguments.
    """

    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(root_dir, "out"),
        help=(
            "Path to output directory. "
            "A new one will be created if the directory does not exist."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help=(
            "Name of the current experiment."
            "Checkpoints will be stored in '{out_dir}/{name}/ckpt/'. "
            "Logs will be stored in '{out_dir}/{name}/log/'. "
            "If there are existing checkpoints in '{out_dir}/{name}/ckpt/', "
            "training will resume from the last checkpoint."
        ),
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help=(
            "Resumes training using the last checkpoint in '{out_dir}/{name}/ckpt/' if set. "
            "Throws error if '{out_dir}/{name}/' already exists by default."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Manual seed for reproducibility."
    )
    parser.add_argument(
        "--im_size",
        type=int,
        default=28,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during training.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=1500000, help="Number of steps to train for."
    )
    parser.add_argument(
        "--repeat_d",
        type=int,
        default=5,
        help="Number of discriminator updates before a generator update.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help="Number of steps between model evaluation.",
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=5,
        help="Number of steps between checkpointing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to train on.",
    )
    parser.add_argument(
        "--mlp_d_checkpoint",
        type=str,
        default=None,
        help="MLP discriminator model checkpoint"
    )

    parser.add_argument(
        "--ckpt_total",
        type=str,
        default=None,
        help="Total model checkpoint"
    )

    return parser.parse_args()


def train(args):
    r"""
    Configures and trains model.
    """

    # Print command line arguments and architectures
    pprint.pprint(vars(args))

    # Setup dataset
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory 'args.data_dir' is not found.")

    # Check existing experiment
    exp_dir = os.path.join(args.out_dir, args.name)
    if os.path.exists(exp_dir) and not args.resume:
        raise FileExistsError(
            f"Directory '{exp_dir}' already exists. "
            "Set '--resume' if you wish to resume training or "
            "change '--name' if you wish to start a new experiment."
        )

    # Setup output directories
    log_dir = os.path.join(exp_dir, "log")
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    for d in [args.out_dir, exp_dir, log_dir, ckpt_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    # Fixed seed
    torch.manual_seed(args.seed)

    # Set parameters
    nz, lr, betas, eval_size, num_workers = (64, 2e-4, (0.0, 0.9), 1000, 4)

    # Configure models

    mlp_discriminator = MLP_Discriminator()
    if args.mlp_d_checkpoint:
        checkpoint = torch.load(args.mlp_d_checkpoint)
        mlp_discriminator.load_state_dict(checkpoint['net_d'])
    classifier = MLP_Classifier(mlp_discriminator = mlp_discriminator)

    # Configure optimizers
    opt_c = optim.Adam(classifier.parameters(), lr, betas)

    # Configure schedulers
    sch_c = optim.lr_scheduler.LambdaLR(
        opt_c, lr_lambda=lambda s: 1.0 - (s / args.max_steps)
    )
    
    # Configure dataloaders
    train_dataloader, eval_dataloader = util.get_dataloaders_MNIST(
        args.data_dir, args.im_size, args.batch_size, eval_size, num_workers
    )

    # Configure trainer
    trainer = Trainer(
        classifier,
        opt_c,
        sch_c,
        train_dataloader,
        eval_dataloader,
        nz,
        log_dir,
        ckpt_dir,
        torch.device(args.device),
    )

    # Train model
    trainer.train(args.max_steps, args.repeat_d, args.eval_every, args.ckpt_every)
    acc = trainer.eval(args.ckpt_total)

    return acc

if __name__ == "__main__":
    train(parse_args())

    # ckpt_dir = "./save_vanilla_easy/checkpoints/"
    # accs = []
    # for epoch in range(40)[::2]:
    #     args = parse_args()
    #     args.mlp_d_checkpoint = os.path.join(ckpt_dir, "{}.pth".format(epoch))
    #     accs += [train(args)]

    # np.save('../vanilla_easy_res.npy', np.array(accs))
    # plt.plot(accs)
    # plt.show()

    # ckpt_dir = "./save_cycle/checkpoints/"
    # accs = []
    # for epoch in range(40)[::2]:
    #     args = parse_args()
    #     args.mlp_d_checkpoint = os.path.join(ckpt_dir, "{}.pth".format(epoch))
    #     accs += [train(args)]

    # np.save('../cycle_res.npy', np.array(accs))
    # plt.plot(accs)
    # plt.show()
