import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tbx
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from model import *
# from torchmetrics import IS, FID, KID
import torchmetrics

import matplotlib.pyplot as plt


def prepare_data_for_inception(x, device):
    r"""
    Preprocess data to be feed into the Inception model.
    """
    # breakpoint()
    x = F.interpolate(x, 299, mode="bicubic", align_corners=False)
    minv, maxv = float(x.min()), float(x.max())
    x.clamp_(min=minv, max=maxv).add_(-minv).div_(maxv - minv + 1e-5)
    x.mul_(255).add_(0.5).clamp_(0, 255)

    return x.to(device).to(torch.uint8)


def prepare_data_for_gan(x, nz, device):
    r"""
    Helper function to prepare inputs for model.
    """

    return (
        x.to(device),
        torch.randn((x.size(0), nz)).to(device),
    )

def bce_loss(preds, labels):
    # bce = torch.nn.BCEWithLogitsLoss()
    return F.cross_entropy(preds, labels)
    # return bce(preds, F.one_hot(labels, num_classes = 10).float()).mean()

def compute_loss_c(net_c, bce_loss, x, labels):
    r"""
    General implementation to compute classifier loss.
    """

    preds = net_c(x)
    loss = bce_loss(preds, labels)
    pred_label = preds.argmax(dim=1)
    correct = (pred_label == labels)
    accuracy = correct.sum()/correct.shape[0]

    return loss, accuracy

def train_step(net, opt, sch, compute_loss):
    r"""
    General implementation to perform a training step.
    """
    opt.zero_grad()
    net.train()
    loss, accuracy = compute_loss()
    net.zero_grad()
    loss.backward()
    # nn.utils.clip_grad_value_(net.parameters(), clip_value=0.1)
    opt.step()
    sch.step()

    return loss, accuracy

def evaluate(net_c, dataloader, nz, device, samples_z=None):
    r"""
    Evaluates model and logs metrics.
    Attributes:
        net_g (Module): Torch generator model.
        net_d (Module): Torch discriminator model.
        dataloader (Dataloader): Torch evaluation set dataloader.
        nz (int): Generator input / noise dimension.
        device (Device): Torch device to perform evaluation on.
        samples_z (Tensor): Noise tensor to generate samples.
    """

    net_c.to(device).eval()

    with torch.no_grad():

        # Initialize metrics
        loss_cs = []
        is_, fid, kid, loss_cs = (
            []
        )

        for data, _ in tqdm(dataloader, desc="Evaluating Model"):

            # Compute losses and save intermediate outputs
            
            loss_c, accuracy = compute_loss_c(
                net_c, 
                bce_loss, 
                x, 
                labels
            )
            

            # Update metrics
            loss_cs.append(loss_c)
            # loss_ds.append(loss_d)
            # real_preds.append(compute_prob(real_pred))
            # fake_preds.append(compute_prob(fake_pred))
            # reals = prepare_data_for_inception(reals, device)
            # fakes = prepare_data_for_inception(fakes, device)
            # is_.update(fakes)
            # fid.update(reals, real=True)
            # fid.update(fakes, real=False)
            # kid.update(reals, real=True)
            # kid.update(fakes, real=False)

        # Process metrics
        metrics = {
            "L(G)": torch.stack(loss_gs).mean().item(),
            "L(D)": torch.stack(loss_ds).mean().item(),
            "D(x)": torch.stack(real_preds).mean().item(),
            "D(G(z))": torch.stack(fake_preds).mean().item(),
            # "IS": is_.compute()[0].item(),
            # "FID": fid.compute().item(),
            # "KID": kid.compute()[0].item(),
        }

        # Create samples
        if samples_z is not None:
            samples = net_g(samples_z)
            samples = F.interpolate(samples, 256).cpu()
            samples = vutils.make_grid(samples, nrow=6, padding=4, normalize=True)

    return metrics if samples_z is None else (metrics, samples)


class Trainer:
    r"""
    Trainer performs GAN training, checkpointing and logging.
    Attributes:
        net_g (Module): Torch generator model.
        net_d (Module): Torch discriminator model.
        opt_g (Optimizer): Torch optimizer for generator.
        opt_d (Optimizer): Torch optimizer for discriminator.
        sch_g (Scheduler): Torch lr scheduler for generator.
        sch_d (Scheduler): Torch lr scheduler for discriminator.
        train_dataloader (Dataloader): Torch training set dataloader.
        eval_dataloader (Dataloader): Torch evaluation set dataloader.
        nz (int): Generator input / noise dimension.
        log_dir (str): Path to store log outputs.
        ckpt_dir (str): Path to store and load checkpoints.
        device (Device): Torch device to perform training on.
    """

    def __init__(
        self,
        classifier,
        opt_c,
        sch_c,
        train_dataloader,
        eval_dataloader,
        nz,
        log_dir,
        ckpt_dir,
        device,
    ):
        # Setup models, dataloader, optimizers
        self.net_c = classifier.to(device)
        self.opt_c = opt_c
        self.sch_c = sch_c
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Setup training parameters
        self.device = device
        self.nz = nz
        self.step = 0

        # Setup checkpointing, evaluation and logging
        self.fixed_z = torch.randn((36, nz), device=device)
        self.logger = tbx.SummaryWriter(log_dir)
        self.ckpt_dir = ckpt_dir

    def _state_dict(self):
        return {
            "net_c": self.net_c.state_dict(),
            "opt_c": self.opt_c.state_dict(),
            "sch_c": self.sch_c.state_dict(),
            "step": self.step,
        }

    def _load_state_dict(self, state_dict):
        self.net_c.load_state_dict(state_dict["net_c"])
        self.opt_c.load_state_dict(state_dict["opt_c"])
        self.sch_c.load_state_dict(state_dict["sch_c"])

        self.step = state_dict["step"]

    def _load_checkpoint(self):
        r"""
        Finds the last checkpoint in ckpt_dir and load states.
        """

        ckpt_paths = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
        # if ckpt_paths:  # Train from scratch if no checkpoints were found
        #     ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
        #     ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
        #     self._load_state_dict(torch.load(ckpt_path))

    def _save_checkpoint(self):
        r"""
        Saves model, optimizer and trainer states.
        """

        ckpt_path = os.path.join(self.ckpt_dir, f"{self.step}.pth")
        torch.save(self._state_dict(), ckpt_path)

    def _log(self, metrics, samples):
        r"""
        Logs metrics and samples to Tensorboard.
        """

        for k, v in metrics.items():
            self.logger.add_scalar(k, v, self.step)
        self.logger.add_image("Samples", samples, self.step)
        self.logger.flush()

    def _train_step_c(self, x, labels):
        r"""
        Performs a classifier training step.
        """

        return train_step(
            self.net_c,
            self.opt_c,
            self.sch_c,
            lambda: compute_loss_c(
                self.net_c, 
                bce_loss, 
                x, 
                labels
            ),
        )

    def train(self, max_steps, repeat_d, eval_every, ckpt_every):
        r"""
        Performs GAN training, checkpointing and logging.
        Attributes:
            max_steps (int): Number of steps before stopping.
            repeat_d (int): Number of discriminator updates before a generator update.
            eval_every (int): Number of steps before logging to Tensorboard.
            ckpt_every (int): Number of steps before checkpointing models.
        """

        self._load_checkpoint()

        accuracies = []

        while True:
            pbar = tqdm(self.train_dataloader)
            for x, y in pbar:

                # Training step
                # reals, z = prepare_data_for_gan(data, self.nz, self.device)
                loss_c, accuracy = self._train_step_c(x, y)

                pbar.set_description(
                    f"L(C):{loss_c.item():.2f}|{self.step}/{max_steps}|Accuracy:{'%.3f' % accuracy}"
                )

                # if self.step != 0 and self.step % eval_every == 0:
                #     self._log(
                #         *evaluate(
                #             self.net_g,
                #             self.net_d,
                #             self.eval_dataloader,
                #             self.nz,
                #             self.device,
                #             samples_z=self.fixed_z,
                #         )
                #     )

                if self.step != 0 and self.step % ckpt_every == 0:
                    self._save_checkpoint()

            accuracies += [accuracy]

            self.step += 1
            if self.step >= 5:
                break

            
            # plt.figure()
            # plt.xlim(0, 6)
            # plt.plot(range(1, 6), d_losses, label='d loss')
            # plt.plot(range(1, 6), g_losses, label='g loss')    
            # plt.legend()
            # plt.savefig(os.path.join(save_dir, 'loss.pdf'))
            # plt.close()

        import numpy as np
        np.save(os.path.join(os.path.join(self.ckpt_dir, 'accuracy.npy')), accuracies)

        plt.figure()
        plt.xlim(0, 6)
        plt.ylim(0, 1)
        plt.plot(range(1, 6), accuracies, label='accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.ckpt_dir, 'accuracy.png'))
        plt.close()

    def eval(self, ckpt):
        r"""
        Performs GAN training, checkpointing and logging.
        Attributes:
            max_steps (int): Number of steps before stopping.
            repeat_d (int): Number of discriminator updates before a generator update.
            eval_every (int): Number of steps before logging to Tensorboard.
            ckpt_every (int): Number of steps before checkpointing models.
        """

        # self._load_checkpoint()
        if ckpt:
            checkpoint = torch.load(ckpt)
            self.net_c = MLP_Classifier()
            self.net_c.load_state_dict(checkpoint['net_d'])
        

        pbar = tqdm(self.eval_dataloader)
        metric = torchmetrics.Accuracy()

        for x, y in pbar:
            preds = self.net_c(x)
            acc = metric(preds, y)
        acc = metric.compute()
        print(f"Accuracy: {acc}")
        return acc
