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

def run_stage2_trainer(train_loader, G2, D2, D2_4x4, optimizer_G2, optimizer_D2,
                       optimizer_D2_4x4, args):

    batch_size = args.batchSize

    real_label = 1
    fake_label = 0

    device = torch.device("cuda:0" if args.cuda else "cpu")

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
    criterion_L1 = nn.L1Loss()

    width = args.Stage2imageSize 
    height = args.Stage2imageSize
    channels = args.nc

    if args.cuda:
        criterion_BCE = criterion_BCE.cuda()
        criterion_MSE = criterion_MSE.cuda()
        criterion_L1 = criterion_L1.cuda()

    for epoch in range(100):
        for i, (src, tgt) in enumerate(train_loader):
            if i==16000:
                break

            src = Variable(src)
            src = src.cuda()

            tgt = Variable(tgt)
            tgt = tgt.cuda()
            
            label = torch.full((batch_size,), real_label, device=device)
            label_4x4 = torch.full((batch_size,4,4), real_label, device=device)

            for p in G2.parameters():
                p.requires_grad = True

            for p in D2.parameters():
                p.requires_grad = False

            G2.zero_grad()

            #z = torch.FloatTensor(args.batchSize, args.nz).normal_(0,1)

            #if args.cuda:
            #    z = z.cuda()
            
            #z = Variable(z)
            #fake1 = G1(z)

            fake = G2(src)
            D2_fake = D2(fake)
            D2_4x4_fake = D2_4x4(fake)


            #Supervised (L1) loss
            L1_loss = 0.5*criterion_L1(fake, tgt)
            #L1_loss *= 1.0
            L1_loss.backward(retain_graph=True)

            #fill with label '1'
            label.fill_(real_label)
            label_4x4.fill_(real_label)
            
            #Global Adversarial Loss
            #G2_loss = criterion_BCE(D2_fake, label)
            G2_loss = criterion_MSE(D2_fake, label)
            G2_loss *= 0.025
            G2_loss.backward(retain_graph=True)

            #Patch Adversarial Loss
            #G2_loss_4x4 = criterion_BCE(D2_4x4_fake, label_4x4)
            G2_loss_4x4 = criterion_MSE(D2_4x4_fake, label_4x4)
            G2_loss_4x4 *= 0.025
            #G2_loss_4x4 = 0.025*criterion_MSE(D2_4x4_fake, label_4x4)
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
            D2_real = D2(tgt)
            label.fill_(real_label)

            #D2_loss_real = criterion_BCE(D2_real, label)
            D2_loss_real = criterion_MSE(D2_real, label)
            D2_loss_real.backward(retain_graph=True)

            D2_4x4_real = D2_4x4(tgt)
            label_4x4.fill_(real_label)
            #D2_loss_4x4_real = criterion_BCE(D2_4x4_real, label_4x4)
            D2_loss_4x4_real = criterion_MSE(D2_4x4_real, label_4x4)
            D2_loss_4x4_real.backward()

            #Disregard z's
            #D2(G2(G1(z))) - fake
            #z = torch.FloatTensor(args.batchSize, args.nz).normal_(0,1)
            
            #if args.cuda:
            #    z = z.cuda()
            
            #z = Variable(z)
            
            #fake1 = G1(z)

            fake = G2(src)
            D_fake = D2(fake.detach())
            label.fill_(fake_label)

            #D2_loss_fake = criterion_BCE(D_fake, label)
            D2_loss_fake = criterion_MSE(D_fake, label)
            D2_loss_fake.backward(retain_graph=True)

            D_fake_4x4 = D2_4x4(fake.detach())
            label_4x4.fill_(fake_label)
            #D2_loss_4x4_fake = criterion_BCE(D_fake_4x4, label_4x4)
            D2_loss_4x4_fake = criterion_MSE(D_fake_4x4, label_4x4)
            D2_loss_4x4_fake.backward()

            optimizer_D2.step()
            optimizer_D2_4x4.step()

            if i %100 == 0:
                print('saving images for batch', i)
                save_image(src.squeeze().data.cpu().detach(), 'source.png')
                save_image(tgt.squeeze().data.cpu().detach(), 'target.png')
                save_image(fake.squeeze().data.cpu().detach(), 'fake.png')

            if i % 100 == 0:
                torch.save(G2, './G2_model.pt')
                torch.save(D2, './D2_model.pt')
                torch.save(D2_4x4, './D2_4x4_model.pt')
                
                print('%d [%d/%d] G Loss [L1/GAdv/PAdv] [%.4f/%.4f/%.4f] Loss D (real/fake/fake4x4) [%.4f/%.4f/%.4f]'%
                      (epoch, i, len(train_loader), L1_loss,
                       G2_loss, G2_loss_4x4, D2_loss_real, D2_loss_fake, D2_loss_4x4_fake))


                
                
    
