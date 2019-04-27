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

from utils import weights_init_G
from utils import weights_init_D

def run_stage1_trainer(train_loader, G1, D1, optimizer_G1, optimizer_D1, G1_scheduler, 
                       D1_scheduler, args):

    batch_size = args.batchSize
    
    real_label = 1
    fake_label = 0

    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.restart == '':
        G1.apply(weights_init_G)
        D1.apply(weights_init_D)
    else:
        G1 = torch.load('./G1_model.pt')
        D1 = torch.load('./D1_model.pt')

    criterion_MSE = nn.MSELoss()
    criterion_BCE = nn.BCELoss()

    if args.cuda:
        criterion_BCE = criterion_BCE.cuda()

    for epoch in range(100):
        for i, (images, labels) in enumerate(train_loader):
            if i==16000:
                break

            images = Variable(images)
            images = images.cuda()
            label = torch.full((batch_size,), real_label, device=device)

            #train net_G
            for p in G1.parameters():
                p.requires_grad = True

            for p in D1.parameters():
                p.requires_grad = False

            G1.zero_grad()

            #G_loss
            z = torch.FloatTensor(args.batchSize, args.nz).normal_(0, 1)
            #z = torch.randn(args.batchSize, args.nz)

            if args.cuda:
                z = z.cuda()

            z = Variable(z)
            
            fake1 = G1(z)
            D1_fake = D1(fake1)
            
            #fake2 = G2(fake1)
            #D2_fake = D2(fake2)
            
            #features_real, D_real_GAN = net_D(images)

            #fill with label '1'
            label.fill_(real_label)

            #G1_loss = criterion_MSE(D1_fake, label)
            G1_loss = criterion_BCE(D1_fake, label)
            G1_loss.backward(retain_graph=True)

            optimizer_G1.step()


            #train net_D
            for p in D1.parameters():
                p.requires_grad = True

            for p in G1.parameters():
                p.requires_grad = False

            D1.zero_grad()

            #real
            D1_real = D1(images)

            label.fill_(real_label)

            #D_loss_real = criterion_BCE(D_real, label)
            #D1_loss_real = criterion_MSE(D1_real, label)

            D1_loss_real = criterion_BCE(D1_real, label)
            D1_loss_real.backward(retain_graph=True)

            #fake
            z = torch.FloatTensor(args.batchSize, args.nz).normal_(0, 1)
            #z = torch.randn(args.batchSize, args.nz)

            if args.cuda:
                z = z.cuda()

            z = Variable(z)

            fake1 = G1(z)
            D_fake1 = D1(fake1.detach())
            
            #fill with label '0'
            label.fill_(fake_label)

            #D1_loss_fake = criterion_MSE(D_fake1, label)
            D1_loss_fake = criterion_BCE(D_fake1, label)
            D1_loss_fake.backward(retain_graph=True)

            optimizer_D1.step()

            if  i % 100 == 0 :
                print('saving images for batch', i)
                save_image(fake1.squeeze().data.cpu().detach(), './fake1.png')
                save_image(images.data.cpu().detach(), './real1.png')

            if i % 100 == 0:
                torch.save(G1, './G1_model.pt')
                torch.save(D1, './D1_model.pt')
                
                print('%d [%d/%d] Loss G %.4f Loss D (real/fake) [%.4f/%.4f]'%
                      (epoch, i, len(train_loader), 
                       G1_loss, D1_loss_real, D1_loss_fake))
