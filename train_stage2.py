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

def run_stage2_trainer(train_loader, G1, G2, D2, D2_4x4, optimizer_G2, optimizer_D2,
                       optimizer_D2_4x4, args):

    batch_size = args.batchSize

    real_label = 1
    fake_label = 0

    device = torch.device("cuda:0" if args.cuda else "cpu")

    G1 = torch.load('./G1_model.pt')

    if args.restart == '':
        G2.apply(weights_init_G)
        D2.apply(weights_init_D)
        D2_4x4.apply(weights_init_D)
    else:
        G2 = torch.load('./G2_model.pt')
        D2 = torch.load('./D2_model.pt')
        D2_4x4 = torch.load('./D2_4x4_model.pt')



    criterion_BCE = nn.BCELoss()
    criterion_MSE = nn.MSELoss()

    if args.cuda:
        criterion_BCE = criterion_BCE.cuda()
        criterion_MSE = criterion_MSE.cuda()

    #freeze G1 just in case    
    for p in G1.parameters():
        p.requires_grad = False

    for epoch in range(100):
        for i, (images, labels) in enumerate(train_loader):
            if i==16000:
                break

            images = Variable(images)
            images = images.cuda()
            label = torch.full((batch_size,), real_label, device=device)
            label_4x4 = torch.full((batch_size,4,4), real_label, device=device)

            for p in G2.parameters():
                p.requires_grad = True

            for p in D2.parameters():
                p.requires_grad = False

            G2.zero_grad()

            z = torch.FloatTensor(args.batchSize, args.nz).normal_(0,1)

            if args.cuda:
                z = z.cuda()
            
            z = Variable(z)
            fake1 = G1(z)
            
            fake2 = G2(fake1)
            D2_fake = D2(fake2)
            D2_4x4_fake = D2_4x4(fake2)

            #fill with label '1'
            label.fill_(real_label)
            label_4x4.fill_(real_label)
           
            #G2_loss = criterion_BCE(D2_fake, label)
            G2_loss = criterion_MSE(D2_fake, label)
            G2_loss.backward(retain_graph=True)

            #G2_loss_4x4 = criterion_BCE(D2_4x4_fake, label_4x4)
            G2_loss_4x4 = criterion_MSE(D2_4x4_fake, label_4x4)
            G2_loss_4x4.backward()
            
            optimizer_G2.step()

            #train D2
            for p in D2.parameters():
                p.requires_grad = True
                
            for p in G2.parameters():
                p.requires_grad = False

            D2.zero_grad()
            D2_4x4.zero_grad()
            
            #real 
            D2_real = D2(images)
            label.fill_(real_label)

            #D2_loss_real = criterion_BCE(D2_real, label)
            D2_loss_real = criterion_MSE(D2_real, label)
            D2_loss_real.backward(retain_graph=True)

            D2_4x4_real = D2_4x4(images)
            label_4x4.fill_(real_label)
            #D2_loss_4x4_real = criterion_BCE(D2_4x4_real, label_4x4)
            D2_loss_4x4_real = criterion_MSE(D2_4x4_real, label_4x4)
            D2_loss_4x4_real.backward()

            #D2(G2(G1(z))) - fake
            z = torch.FloatTensor(args.batchSize, args.nz).normal_(0,1)

            if args.cuda:
                z = z.cuda()
            
            z = Variable(z)
            
            fake1 = G1(z)

            fake2 = G2(fake1)
            D_fake2 = D2(fake2.detach())
            label.fill_(fake_label)

            #D2_loss_fake = criterion_BCE(D_fake2, label)
            D2_loss_fake = criterion_MSE(D_fake2, label)
            D2_loss_fake.backward(retain_graph=True)

            D_fake2_4x4 = D2_4x4(fake2.detach())
            label_4x4.fill_(fake_label)
            #D2_loss_4x4_fake = criterion_BCE(D_fake2_4x4, label_4x4)
            D2_loss_4x4_fake = criterion_MSE(D_fake2_4x4, label_4x4)
            D2_loss_4x4_fake.backward()

            optimizer_D2.step()
            optimizer_D2_4x4.step()

            if i %100 == 0:
                print('saving images for batch', i)
                save_image(fake1.squeeze().data.cpu().detach(), 'fake1_2.png')
                save_image(fake2.squeeze().data.cpu().detach(), './fake2.png')
                save_image(images.data.cpu().detach(), './real2.png')

            if i % 100 == 0:
                torch.save(G2, './G2_model.pt')
                torch.save(D2, './D2_model.pt')
                
                print('%d [%d/%d] Loss G %.4f Loss D (real/fake/fake4x4) [%.4f/%.4f/%.4f]'%
                      (epoch, i, len(train_loader), 
                       G2_loss, D2_loss_real, D2_loss_fake, D2_loss_4x4_fake))


                
                
    
