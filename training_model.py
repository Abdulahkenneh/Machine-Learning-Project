import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator Model
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
noise_dim = 100
img_dim = 64 * 64 * 3  # Assuming the images are 64x64 RGB images
lr = 0.0002
batch_size = 64
epochs = 100

# Initialize generator and discriminator
generator = Generator(noise_dim, img_dim).to(device)
discriminator = Discriminator(img_dim).to(device)

# Optimizers
optim_gen = optim.Adam(generator.parameters(), lr=lr)
optim_disc = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

# DataLoader for training images
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(root='path_to_your_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training Loop
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.view(-1, img_dim).to(device)
        batch_size = real.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake = generator(noise)

        disc_real = discriminator(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        optim_disc.zero_grad()
        loss_disc.backward()
        optim_disc.step()

        # Train Generator
        output = discriminator(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

# Save models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
