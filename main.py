import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np
from tqdm import tqdm

from model import *

# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 28 # 784
num_epochs = 150
batch_size = 32

# experiment_condition = 'vanilla'
# experiment_condition = 'vanilla_easy'
experiment_condition = 'cycle'

save_dir = 'save_' + experiment_condition
ckpt_dir = os.path.join(save_dir, 'checkpoints')
sample_dir = 'samples_' + experiment_condition

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(ckpt_dir)

# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5),   # 3 for RGB channels
                                     std=(0.5))])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)

# Discriminator
D = MLP_Discriminator(image_size, latent_size, hidden_size)

# Generator
G = MLP_Generator(image_size, latent_size, hidden_size)

# Device setting
D = D
G = G

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

def get_state_dict(epoch, D, G, g_optimizer, d_optimizer):
    return {
        "net_g": G.state_dict(),
        "net_d": D.state_dict(),
        "opt_g": g_optimizer.state_dict(),
        "opt_d": d_optimizer.state_dict(),
    }

# Statistics to be saved
d_losses = np.zeros(num_epochs)
g_losses = np.zeros(num_epochs)
real_scores = np.zeros(num_epochs)
fake_scores = np.zeros(num_epochs)

# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    pbar = tqdm(data_loader)
    for i, (images, _) in enumerate(pbar):

        images = images.view(batch_size, -1)
        images = Variable(images)
        d_inputs = images[:batch_size//2,:]
        g_inputs = images[batch_size//2:,:]
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size//2, 1)
        real_labels = Variable(real_labels)
        fake_labels = torch.zeros(batch_size//2, 1)
        fake_labels = Variable(fake_labels)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs, _ = D(d_inputs)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        if 'vanilla' in experiment_condition:
            z = torch.randn(batch_size, latent_size)
            z = Variable(z)
        elif 'cycle' in experiment_condition:
            _, z = D(g_inputs)
        else:
            breakpoint()
        fake_images = G(z)
        outputs, _ = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        # If D is trained so well, then don't update
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size)
        z = Variable(z)
        fake_images = G(z)
        outputs, _ = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        # if G is trained so well, then don't update
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        # =================================================================== #
        #                          Update Statistics                          #
        # =================================================================== #
        d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.data.item()*(1./(i+1.))
        g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.data.item()*(1./(i+1.))
        real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().data.item()*(1./(i+1.))
        fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().data.item()*(1./(i+1.))
        
        pbar.set_description(
            'Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                .format(epoch, num_epochs, i+1, total_step, d_loss.data.item(), g_loss.data.item(), 
                    real_score.mean().data.item(), fake_score.mean().data.item())
        )

    # Save real images
    if (epoch+1) == 1:
        images = images.view(images.size(0), 1, 28, 28)
        save_image(denorm(images.data), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))
    
    # Save and plot Statistics
    np.save(os.path.join(save_dir, 'd_losses.npy'), d_losses)
    np.save(os.path.join(save_dir, 'g_losses.npy'), g_losses)
    np.save(os.path.join(save_dir, 'fake_scores.npy'), fake_scores)
    np.save(os.path.join(save_dir, 'real_scores.npy'), real_scores)
    
    plt.figure()
    pylab.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), d_losses, label='d loss')
    plt.plot(range(1, num_epochs + 1), g_losses, label='g loss')    
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.pdf'))
    plt.close()

    plt.figure()
    pylab.xlim(0, num_epochs + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, num_epochs + 1), fake_scores, label='fake score')
    plt.plot(range(1, num_epochs + 1), real_scores, label='real score')    
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy.pdf'))
    plt.close()

    if (epoch) % 20 == 0:
        ckpt_path = os.path.join(ckpt_dir, f"{epoch}.pth")
        torch.save(get_state_dict(epoch, D, G, g_optimizer, d_optimizer), ckpt_path)
        # torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G--{}.ckpt'.format(epoch+1)))
        # torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D--{}.ckpt'.format(epoch+1)))

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')