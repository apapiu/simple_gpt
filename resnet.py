import torch
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c_in, c_mid, 3, padding='same'),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(c_mid),
                                  nn.Conv2d(c_mid, c_out, 3, padding='same'),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(c_out)
                                  )
        
        self.resid = nn.Conv2d(c_in, c_out, 1, padding='same') if c_in != c_out else nn.Identity()
    
    def forward(self, x):

        return self.conv(x) + self.resid(x)

class ResTower(nn.Module):
    def __init__(self, c_in, img_dim):
        super().__init__()
        self.tower = nn.Sequential(ResBlock(3,      1*c_in, 1*c_in),
                                   ResBlock(1*c_in, 1*c_in, 1*c_in),
                                   nn.MaxPool2d(2),
                                   ResBlock(1*c_in, 2*c_in, 2*c_in),
                                   ResBlock(2*c_in, 2*c_in, 2*c_in),
                                   nn.MaxPool2d(2),
                                   ResBlock(2*c_in, 4*c_in, 4*c_in),
                                   ResBlock(4*c_in, 4*c_in, 8*c_in),
                                   nn.AvgPool2d(img_dim//4), #512 dim for 32 input
                                   nn.Flatten(),
                                   nn.Linear(8*c_in, 4*c_in),
                                   nn.ReLU(),
                                   nn.Linear(4*c_in, 10),
                                   )
                                   
        
    def forward(self, x):
         return self.tower(x) 


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = ResTower(64, 32)

        self.global_step = 0
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(lr=0.0003, params=self.parameters(), weight_decay = 1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=12000)

    def forward(self, x):
        return self.tower(x)

    def train_step(self, x, y):

        self.optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1
        return loss.item()

    def validate(self):
            accus = []
            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    self.eval()
                    preds = self(x)
                    self.train()
                preds = preds.argmax(1)
                acc = (preds == y).float().mean()
                accus.append(acc.item())
            print(f'val accu: {np.array(accus).mean()} step: {self.global_step}')
            print(f'learning rate: {model.scheduler.get_last_lr()}')

model = Model().to(device)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(10),       # Randomly rotate the image by a maximum of 10 degrees
    transforms.RandomCrop(32, padding=4), # Randomly crop the image with padding of 4 pixels
    transforms.ToTensor(),
])

transform2 = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

# Load CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform2)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=2)

n_epochs = 30

for i in range(n_epochs):
    for x, y in tqdm(trainloader):
        if model.global_step % 200 == 0:
            model.validate()
        model.train_step(x, y)
