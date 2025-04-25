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

path = 'C:/Users/MSI/Desktop/datasets/Rice_Image_Dataset'

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

train_dataload = DataLoader(
    dataset=path,
    batch_size=128,
    shuffle=True
    )

val_dataload = DataLoader(
    dataset=path,
    batch_size=128,
    shuffle=True
    )

class CustomDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.classes = os.listdir(path)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.path, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            images.append((image_path, self.class_to_idx[class_name]))
        return images           

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

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