import os
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image

train_transform = transforms.Compose([
    transforms.CenterCrop(225),
    transforms.RandomHorizontalFlip(p=0.45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.CenterCrop(230),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.Conv2d(),
            nn.ReLU(),
            nn.Linear(),
            nn.Linear()
        )

    def forward(self, x):
        return self.layers(x)

model = MyNet()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD()