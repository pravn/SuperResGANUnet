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



def run_trainer1(train_loader, G1, D1, args):
    from train_stage1 import run_stage1_trainer
    optimizer_G1 = optim.Adam(G1.parameters(), lr=args.lr, betas=(args.beta1,0.999))
    G1_scheduler = StepLR(optimizer_G1, step_size=1000, gamma=1.0)
    
    optimizer_D1 = optim.Adam(D1.parameters(), lr=args.lr, betas=(args.beta1,0.999))
    D1_scheduler = StepLR(optimizer_D1, step_size=1000, gamma=1.0)

    run_stage1_trainer(train_loader, G1, D1, optimizer_G1, optimizer_D1, G1_scheduler, 
                       D1_scheduler, args)


