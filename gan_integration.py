import torch
import torch.nn as nn
import pygame
import numpy as np
from perlin_noise import PerlinNoise
from torchvision.transforms import ToPILImage 
import PIL.Image
import torch.nn.functional as F

# Параметры GAN
LATENT_DIM = 100
CHANNELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Генератор с Residual Blocks для 128x128 на входе, 64x64 на выходе
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))) 
        shortcut = self.shortcut(x)
        
        if out.size(2) != shortcut.size(2) or out.size(3) != shortcut.size(3):
            shortcut = F.interpolate(shortcut, size=out.size()[2:], mode='bilinear', align_corners=False)
        
        return out + shortcut
    
class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.1):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:  # Шум добавляется только во время тренировки
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x

# Генератор с Residual Blocks
class GeneratorWithResiduals(nn.Module):
    def __init__(self, latent_dim, channels):
        super(GeneratorWithResiduals, self).__init__()
        self.noise_layer = GaussianNoise(0.05) 
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),  # Из латентного пространства в 4x4x256
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            ResidualBlock(256, 128),
            ResidualBlock(128, 64),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),  # 64x64x3
            nn.Tanh()  # Tanh, чтобы привести значения пикселей к диапазону [-1, 1]
        )

    def forward(self, input):
        input = self.noise_layer(input)  # Добавляем шум
        return self.main(input)


# Инициализация генератора
generator = GeneratorWithResiduals(LATENT_DIM, CHANNELS).to(DEVICE)
generator.load_state_dict(torch.load('C:\\base\\prog\\diplom\\diplom\\PerlinAndGan\\generator.pth', map_location=DEVICE, weights_only=True))

generator.eval()

# Функция генерации текстуры травы с использованием GAN
def generate_grass_texture():
    with torch.no_grad():
        # Генерация шума
        noise = torch.randn(1, LATENT_DIM, 1, 1, device=DEVICE)
        print("Noise shape:", noise.shape)  # Покажем форму шума

        # Генерация изображения
        generated_image = generator(noise)
        print("Generated image shape:", generated_image.shape)  # Покажем форму сгенерированного изображения

        # Преобразование в формат [0, 1] и затем [0, 255] для отображения в Pygame
        generated_image = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
        print("Final generated image shape:", generated_image.shape)  # Проверим окончательный размер

        # Масштабируем изображение до 64x64
        resized_image = np.array(PIL.Image.fromarray((generated_image * 255).astype(np.uint8)).resize((64, 64)))
        print("Resized image shape:", resized_image.shape)  # Проверим форму после изменения размера

        return resized_image

