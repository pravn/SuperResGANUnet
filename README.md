# SuperResGANUnet
Attempt at creating larger images with the StackGAN concept. 
One way of producing higher res images with GANs is to start with a lower resolution image that the GAN is capable of generating (64x64), and then iteratively superresolve that. The idea harks back to the StackGAN paper of successively adding detail to a low res image by treating GANs as a style loss. Nevertheless, from another standpoint, we acknowledge that GANs are incapable of generating high res images without considerable hair pulling. The recent demonstrations in producing high res GAN images (BigGAN, ProgGAN, etc.) are feats that don't seem to be amenable to everyday use for various reasons, not least of them being the computational requirements. Also, it strikes me that most of the good GAN use cases seem to come in scenarios where it functions as a style loss, with content loss coming from conditioning on another image in the form of a supervised component - usually, an L1 loss. 

To that end, we set up our problem as starting with a lower resolution image which serves as the conditioning input for the higher resolution image, and then using the GAN losses as polishing mechanism. 

This is essentially an interpretation of the pix2pix idea - in fact, one could very well use the same code. Nevertheless, we pull the components apart to suit our own needs concerning instruction and edification. 

For this exercise, we create two dataloaders - one each for the low and high res image sets. Then, we create a Unet generator from the Pix2Pix/CycleGAN code base. 

Then we train to produce the higher reesolution image. 

Losses:
Generator
1) Content: L1(generated, target) 
2) Style: lambda * adversarial loss 

Discriminator:
This is a normal DCGAN (or in our case, LSGAN) 0/1 loss with minimal changes from the pytorch DCGAN code. 

Modifications to improve GAN training: Replace the generator loss with a feature loss (like in Salimans 2016 and the VAEGAN paper). This loss is as follows.


\begin{equation}
G_{adv} = || E_{x\sim real} D(real_features) - E_{x\sim fake} D(fake_features)||_2   
\end{equation}

I plan to include other components from the Salimans paper in due season. These include, notably, feature matching (already noted above), mini batch discrimination, which seems to be particularly emphasized, and the idea of using a replay buffer/historical averaging for the discriminator, so as to reduce instability as might arise when the parameters/data for the discriminator get updated more rapidly than stable training might admit. 

PatchGAN: In the Pix2Pix/CycleGAN works, the discriminator (in effect) classifies several small patches of the larger, main image. After reasoning out, we see that this translates to stopping the downsampling process in the discriminator at an earlier stage, instead of going all the way down to a 1x1 classifier output. For example, if we stop at 4x4 (say before applying the operation Conv2d(8 * ndf, 4, 1, 0)), then it means that the discriminator has classified the image into regions corresponding to a grid of 4x4 patches. This lends itself to the interpretation that the patchgan works on a smaller, fine grained scale compared with the global discriminator, yielding perhaps an improvement in fine grained details corresponding to that scale. As we can see, this concept is easy to implement as a regular convolutional discriminator. After obtaining this grid of classified output, we compute the loss using a label grid of the same size, indicating whether or not the obtained classified output comes from the actual image or from the fake distribution. 
