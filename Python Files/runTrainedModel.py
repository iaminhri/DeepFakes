import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# change to 'cuda:0' or 'cpu' or 'mps' if mac.
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

# generator path
generator_path = '/Users/hridoyrahman/Desktop/4P80_Project/savedModel/generator128.pth'
discriminator_path = '/Users/hridoyrahman/Desktop/4P80_Project/savedModel/discriminator128.pth'

# optimizers path
g_optimizer_path = '/Users/hridoyrahman/Desktop/4P80_Project/savedModel/g_optimizer128.pth'
d_optimizer_path = '/Users/hridoyrahman/Desktop/4P80_Project/savedModel/d_optimizer128.pth'

# Generator model
class generatorNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.c0 = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False)
    self.c1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False)
    self.c2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False)
    self.c3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False)
    self.c4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False)
    self.c5 = nn.ConvTranspose2d(32, 3, 4, 2, 1, bias = False)

    # batch normalization
    self.bn0 = nn.BatchNorm2d(512)
    self.bn1 = nn.BatchNorm2d(256)
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm2d(64)
    self.bn4 = nn.BatchNorm2d(32)

  def forward(self, x):
    x = F.relu(self.bn0(self.c0(x)))
    x = F.relu(self.bn1(self.c1(x)))
    x = F.relu(self.bn2(self.c2(x)))
    x = F.relu(self.bn3(self.c3(x)))
    x = F.relu(self.bn4(self.c4(x)))

    return torch.tanh(self.c5(x))

# Discriminator model
class discriminatorNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.c1 = nn.Conv2d(3, 32, 4, 2, 1, bias = False)
    self.c2 = nn.Conv2d(32, 64, 4, 2, 1, bias = False)
    self.c3 = nn.Conv2d(64, 128, 4, 2, 1, bias = False)
    self.c4 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
    self.c5 = nn.Conv2d(256, 512, 4, 2, 1, bias = False)
    self.c6 = nn.Conv2d(512, 1, 4, 1, 0, bias = False)

    # batch Normalization
    self.bn2 = nn.BatchNorm2d(64)
    self.bn3 = nn.BatchNorm2d(128)
    self.bn4 = nn.BatchNorm2d(256)
    self.bn5 = nn.BatchNorm2d(512)

  def forward(self, x):
    x = F.leaky_relu(self.c1(x), 0.2, inplace = True)

    x = F.leaky_relu(self.c2(x), 0.2, inplace = True)
    x = self.bn2(x)

    x = F.leaky_relu(self.c3(x), 0.2, inplace = True)
    x = self.bn3(x)

    x = F.leaky_relu(self.c4(x), 0.2, inplace = True)
    x = self.bn4(x)

    x = F.leaky_relu(self.c5(x), 0.2, inplace = True)
    x = self.bn5(x)

    return torch.sigmoid(self.c6(x)).view(-1,1)

# Loading the models
gNet = generatorNet().to(device)  # Define architecture
dNet = discriminatorNet().to(device)  # Define architecture

# loading optimizers
dOpt = optim.Adam(dNet.parameters(), lr = 0.0002, betas = (0.5, 0.999))
gOpt = optim.Adam(gNet.parameters(), lr = 0.0002, betas = (0.5, 0.9))

# loading the trained models.
gNet.load_state_dict(torch.load(generator_path, map_location=device))
dNet.load_state_dict(torch.load(discriminator_path, map_location=device))

# Loading the optimizers state
gOpt.load_state_dict(torch.load(g_optimizer_path, map_location=device))
dOpt.load_state_dict(torch.load(d_optimizer_path, map_location=device))

# Generating fake images from trained generator model and plotting. 
gNet.eval()
noise = torch.randn(32, 100, 1, 1, device=device)
with torch.no_grad():
    fake_image = gNet(noise).detach().cpu()

import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Converting the tensors to images
fake_image = fake_image.squeeze(0)  # Remove batch dimension
fake_image = vutils.make_grid(fake_image, normalize=True)

# Displaying the images
plt.figure(figsize=(10, 10))
plt.imshow(fake_image.permute(1, 2, 0))  # Rearranging dimensions for matplotlib
plt.axis('off')
plt.show()