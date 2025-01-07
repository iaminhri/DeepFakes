# -*- coding: utf-8 -*-
"""
35KSamples_128x128_DeepFakesGANs.ipynb
Automatically generated by Colab.

"""

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


torch.use_deterministic_algorithms(True)

# change to 'cuda:0' or 'cpu' or 'mps' if mac.
# device = 'mps'
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# generator path
generator_path = '/Users/hridoyrahman/Desktop/4P80_Project/SavedModel/generator128.pth'
discriminator_path = '/Users/hridoyrahman/Desktop/4P80_Project/SavedModel/discriminator128.pth'

# optimizers path
g_optimizer_path = '/Users/hridoyrahman/Desktop/4P80_Project/SavedModel/g_optimizer128.pth'
d_optimizer_path = '/Users/hridoyrahman/Desktop/4P80_Project/SavedModel/d_optimizer128.pth'

# dataset root directory
dataroot = "/Users/hridoyrahman/Desktop/4P80_Project/data/forgeryNet"

# Transforms the image dataset and converts it to tensors.
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

print(f"Number of images in dataset: {len(dataset)}")

dataset[0][0]

# loads the dataset in batchsize using dataloader. 
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

print(f"Number of batches in dataloader: {len(dataloader)}")

real_batch = next(iter(dataloader))
plt.figure(figsize=(12,12))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:16], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

# Generator net
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
  # forward propagation
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
    
  # forward propagation
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

lossFunc = nn.BCELoss()

real_label = 1
fake_label = 0

dNet = discriminatorNet().to(device)
gNet = generatorNet().to(device)

print(dNet)
print(gNet)

dOpt = optim.Adam(dNet.parameters(), lr = 0.0002, betas = (0.5, 0.999))
gOpt = optim.Adam(gNet.parameters(), lr = 0.0002, betas = (0.5, 0.999))

# Commented out IPython magic to ensure Python compatibility.
num_epochs = 6
losses_G = []
losses_D = []
disDecs = []
iters = 0
img_list = []

# Training the models. 
for epoch in range(num_epochs):
  for i, data in enumerate(dataloader, 0):
    dNet.zero_grad()

    # Training with Real Data
    # Training Discriminator Network
    dataSegment = data[0].to(device)
    batchSize = dataSegment.size(0)

    label = torch.full((batchSize,), real_label, dtype=torch.float, device=device)

    output = dNet(dataSegment).view(-1)

    # Calculate loss for discriminator net
    loss_real = lossFunc(output, label)
    loss_real.backward() # Back propagate with respect to loss
    meanLoss_real_D = output.mean().item()

    # Training Generator Network / Training with Fake Data
    noise = torch.randn(batchSize, 100, 1, 1, device = device)
    fakeData = gNet(noise)
    label.fill_(fake_label) # fills label with 0s

    # Classify fake data
    output = dNet(fakeData.detach()).view(-1)

    # Calculate loss of the Generator Net
    loss_fake = lossFunc(output, label)
    loss_fake.backward() # backpropagate

    meanLoss_fake_G = output.mean().item() # mean loss
    loss_total_D = loss_real + loss_fake # Total discriminator loss
    dOpt.step() # update biases and weights

    # Training with Fake Data
    # Generate batch of latent vectors
    gNet.zero_grad()
    label.fill_(real_label) # fills with 1s
    output = dNet(fakeData).view(-1)
    lossG = lossFunc(output, label)
    lossG.backward() # Back Propagate
    meanLoss_real_G = output.mean().item()
    gOpt.step()

    if i % 50 == 0:
      print(f"Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(dataloader)}]")
      print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {loss_total_D.item():.4f}\tLoss_G: {lossG.item():.4f}\tD(x): {meanLoss_real_D:.4f}\tD(G(z)): {meanLoss_fake_G:.4f} / {meanLoss_real_G:.4f}')

    # Save Losses for plotting later
    losses_G.append(lossG.item())
    losses_D.append(loss_total_D.item())

        # Check how the generator is doing by saving G's output on fixed_noise
    if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
      with torch.no_grad():
          fake = gNet(noise).detach().cpu()
      img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    iters += 1


# Save the models
torch.save(gNet.state_dict(), generator_path)
torch.save(dNet.state_dict(), discriminator_path)

torch.save(gOpt.state_dict(), g_optimizer_path)
torch.save(dOpt.state_dict(), d_optimizer_path)

# Load the models
gNet = generatorNet()  # Define architecture
dNet = discriminatorNet()  # Define architecture

gNet.load_state_dict(torch.load(generator_path))
dNet.load_state_dict(torch.load(discriminator_path))

gNet.to(device)
dNet.to(device)

# Load the optimizers
gOpt.load_state_dict(torch.load(g_optimizer_path))
dOpt.load_state_dict(torch.load(d_optimizer_path))

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(losses_G,label="G")
plt.plot(losses_D,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

gNet.eval()
noise = torch.randn(32, 100, 1, 1, device=device)  # Batch size = 1
with torch.no_grad():
    fake_image = gNet(noise).detach().cpu()

import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Convert tensor to image format
fake_image = fake_image.squeeze(0)  # Remove batch dimension
fake_image = vutils.make_grid(fake_image, normalize=True)

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(fake_image.permute(1, 2, 0))  # Rearrange dimensions for matplotlib
plt.axis('off')
plt.show()