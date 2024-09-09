import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils import data
from PIL import Image

# Dataset definition
class CustomImageDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 'L' for grayscale
        if self.transform:
            image = self.transform(image)
        return image
    


class Discriminator(nn.Module):
    def __init__(self, ngpu, channels_img=1, features_d=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input: N x channels_img x 256 x 256
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=4, padding=1),  # Faster downsampling
            nn.LeakyReLU(0.2),
            # State: 64x64
            self._block(features_d, features_d * 2, 4, 4, 1),  # Faster downsampling
            # State: 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # Standard downsampling
            # State: 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # Standard downsampling
            # State: 4x4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0),  # Final output (1, 1, 1)
            nn.Sigmoid()  # Output between 0 and 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.main(x).view(-1)



class Generator(nn.Module):
    def __init__(self, ngpu, channels_noise=100, channels_img=1, features_g=32):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 32, 4, 1, 0),  # img: 4x4
            self._block(features_g * 32, features_g * 16, 4, 2, 1),  # img: 8x8
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 16x16
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 32x32
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 64x64
            self._block(features_g * 2, features_g, 4, 2, 1),  # img: 128x128
            nn.ConvTranspose2d(
                features_g, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 256 x 256
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# Weights initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(batch_size, lr1, lr2, beta1):
    # Hyperparameters
    dataroot = '/home/groups/comp3710/OASIS/keras_png_slices_train/'
    workers = 4
    image_size = 256
    nc = 1  # Number of channels in grayscale images
    nz = 100  # Size of z latent vector
    num_epochs = 50
    ngpu = 1

    # Transform and Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CustomImageDataset(root_dir=dataroot, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers)

    # Device setup
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the Generator and Discriminator
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

    # Loss function
    criterion = nn.BCELoss()

    # Setup optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr1, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr2, betas=(beta1, 0.999))

    # Fixed noise for generating images
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Labels
    real_label = 1.
    fake_label = 0.

    # Training Loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            ## Train with real batch
            netD.zero_grad()
            real_data = data.to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_data = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_data.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = netD(fake_data).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Print training stats
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')


    # Generate and save 7 final fake brains
    with torch.no_grad():
        fake_brains = netG(fixed_noise).detach().cpu()
        save_image(fake_brains[:7], f'brain_{batch_size}_{lr1}_{lr2}_{beta1}.png', normalize=True)

if __name__ == "__main__":
    train(128, 0.0002, 0.0002, 0.5)
