import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
#from torchvision.datasets import MNIST
#from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F


def plot_loss(loss_array,name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_array)
    plt.savefig('loss_'+name)


def run_trainer2(train_loader, G2, D2, D2_4x4, args):
    from train_stage2 import run_stage2_trainer
    optimizer_G2 = optim.Adam(G2.parameters(), lr=args.lr, betas=(args.beta1,0.999))
    G2_scheduler = StepLR(optimizer_G2, step_size=1000, gamma=1.0)
    
    optimizer_D2 = optim.Adam(D2.parameters(), lr=args.lr, betas=(args.beta1,0.999))
    D2_scheduler = StepLR(optimizer_D2, step_size=1000, gamma=1.0)

    optimizer_D2_4x4 = optim.Adam(D2_4x4.parameters(), lr=args.lr, betas=(args.beta1,0.999))

    run_stage2_trainer(train_loader, G2, D2, D2_4x4, optimizer_G2, optimizer_D2, optimizer_D2_4x4,
                       args)
