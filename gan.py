
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import h5py
import pandas as pd

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
latent_dim = 100
image_size = 500
channels = 3  # For RGB images

def data_customize(focus=False):
    # Data transformation and loader
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * channels, [0.5] * channels)
    ])

def data(path):
    ext = os.path.splitext(path)[-1].lower()

    if os.path.isdir(path):
        dataset = datasets.ImageFolder(root=path, transform=data_customize())
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    elif ext == '.h5':
        with h5py.File(path, 'r') as hdf:
            keys = list(hdf.keys())
            data = {key: hdf[key][:] for key in keys}
            return data  
    
    elif ext == '.csv':
        data = pd.read_csv(path)
        return data  
    
    elif ext in ['.jpg', '.jpeg', '.png']:
        dataset = datasets.ImageFolder(root=path, transform=data_customize())
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# Discriminator model using adaptive pooling
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Generator model using adaptive pooling
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Initialize models
generator = Generator()
discriminator = Discriminator()

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
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train Discriminator
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch_size, latent_dim, 1, 1)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    if save:
        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(discriminator.state_dict(), 'discriminator.pth')

# Call the training function
# train(path_to_your_data, num_epochs=10, save=True)
