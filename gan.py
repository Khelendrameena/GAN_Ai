import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import h5py
import pandas as pd

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
latent_dim = 100
image_size = 64
channels = 3  # For RGB images


def data_customize(focus=False):
    # Data transformation and loader
    if focus:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels)
        ])


data_customize(focus=True)

def data(path, focus=False):
    ext = os.path.splitext(path)[-1].lower()

    if os.path.isdir(path):
        # If path is a directory, load image data
        dataset = datasets.ImageFolder(root=path, transform=data_customize(focus))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    elif ext == '.h5':
        # If it's an .h5 file, use h5py to load the data
        with h5py.File(path, 'r') as hdf:
            keys = list(hdf.keys())
            data = {key: hdf[key][:] for key in keys}
            return data  # You might need to return a DataLoader or tensor depending on your use case
    
    elif ext == '.csv':
        # If it's a CSV file, use pandas to load the data
        data = pd.read_csv(path)
        return data  # You might want to convert this to a tensor or DataLoader
    
    elif ext in ['.jpg', '.jpeg', '.png']:
        image_dir = path
        dataset = datasets.ImageFolder(root=image_dir, transform=data_customize(focus))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
# Discriminator model using CNN
class Discriminator(nn.Module):
    def __init__(self,image_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, image_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size, 2*image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*image_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*image_size, 4*image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*image_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*image_size, 8*image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8*image_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8*image_size, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)
        
# Generator model using CNN
class Generator(nn.Module):
    def __init__(self,image_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, image_size*8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(image_size*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size*8, image_size*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size*4, image_size*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size*2, image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Initialize models
generator = Generator(image_size)
discriminator = Discriminator(image_size)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Training the GAN
def train(path, num_epochs=10, save=False):
    dataloader = data(path)
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            # Create real and fake labels
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train Discriminator
            outputs = discriminator(real_images)
            outputs = outputs.view(-1, 1)  # Ensure output size is [batch_size, 1]
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            z = torch.randn(batch_size, latent_dim, 1, 1)  # Note: add (1,1) to match the generator output
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            outputs = outputs.view(-1, 1)  # Ensure output size is [batch_size, 1]
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            d_loss = d_loss_real + d_loss_fake
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            z = torch.randn(batch_size, latent_dim, 1, 1)  # Note: add (1,1) to match the generator output
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            outputs = outputs.view(-1, 1)  # Ensure output size is [batch_size, 1]
            g_loss = criterion(outputs, real_labels)
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, '
                  f'D(x): {real_score.mean().item():.4f}, D(G(z)): {fake_score.mean().item():.4f}')

    if save:
        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(discriminator.state_dict(), 'discriminator.pth')
