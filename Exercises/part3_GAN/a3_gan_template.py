import argparse
import os
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.nn import Sequential, Linear, LeakyReLU, BatchNorm1d, Sigmoid, Softmax, Tanh
from torch.autograd import Variable

CODE_DIR = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/code/part3_GAN/'
WORK_DIR_DEFAULT = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_4/'
IMAGE_DIM = 28


class Generator(nn.Module):
    def __init__(self, latent_dim, n_feat_out, type_activ_out='tanh'):
        super(Generator, self).__init__()
        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        leaky_relu_rate = 0.2

        n_feat_1 = 128
        self.hidden1 = Sequential(Linear(latent_dim, n_feat_1), LeakyReLU(leaky_relu_rate))
        n_feat_2 = 2 * n_feat_1
        self.hidden2 = Sequential(Linear(n_feat_1, n_feat_2), BatchNorm1d(n_feat_2), LeakyReLU(leaky_relu_rate))
        n_feat_3 = 2 * n_feat_2
        self.hidden3 = Sequential(Linear(n_feat_2, n_feat_3), BatchNorm1d(n_feat_3), LeakyReLU(leaky_relu_rate))
        n_feat_4 = 2 * n_feat_3
        self.hidden4 = Sequential(Linear(n_feat_3, n_feat_4), BatchNorm1d(n_feat_4), LeakyReLU(leaky_relu_rate))
        if type_activ_out=='sigmoid':
          self.hidden5 = Sequential(Linear(n_feat_4, n_feat_out), Sigmoid())
        elif type_activ_out=='softmax':
          self.hidden5 = Sequential(Linear(n_feat_4, n_feat_out), Softmax())
        elif type_activ_out=='tanh':
          self.hidden5 = Sequential(Linear(n_feat_4, n_feat_out), Tanh())


    def forward(self, z):
        # Generate images from z

        x = self.hidden1(z)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        out = self.hidden5(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_feat_in, type_activ_out='sigmoid'):
        super(Discriminator, self).__init__()
        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        leaky_relu_rate = 0.2
        n_feat_1 = 512
        self.hidden1 = Sequential(Linear(n_feat_in, n_feat_1), LeakyReLU(leaky_relu_rate))
        n_feat_2 = n_feat_1 / 2
        self.hidden2 = Sequential(Linear(n_feat_1, n_feat_2), LeakyReLU(leaky_relu_rate))
        if type_activ_out=='sigmoid':
          self.hidden3 = Sequential(Linear(n_feat_2, 1), Sigmoid())
        elif type_activ_out=='softmax':
          self.hidden3 = Sequential(Linear(n_feat_2, 1), Softmax())
        elif type_activ_out=='tanh':
          self.hidden3 = Sequential(Linear(n_feat_2, 1), Tanh())


    def forward(self, img):
        # return discriminator score for img
        x = self.hidden1(img)
        x = self.hidden2(x)
        out = self.hidden3(x)
        return out


def generate_Gaussian_noise(size):
    # 1-d vector of gaussian sampled random values
    return Variable(torch.randn(size, args.latent_dim))


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    discriminator.train()
    generator.train()

    loss_BCE = nn.BCELoss()

    def loss_generator(predictions_fake):
        # Compute loss: '-log(D(G(z)))', with z := fake_image
        return loss_BCE(predictions_fake, Variable(torch.ones_like(predictions_fake)))

    def loss_discriminator(predictions_real, predictions_fake):
        # Compute loss: '-log(D(x)) - log(1 - D(G(z)))', with x := real image, z := fake_image
        return loss_BCE(predictions_real, Variable(torch.ones_like(predictions_real))) + \
               loss_BCE(predictions_fake, Variable(torch.zeros_like(predictions_fake)))

    criterion_G = loss_generator
    criterion_D = loss_discriminator

    # file to store loss history
    loss_history_file = os.path.join(args.work_dir, 'loss_history.txt')
    fout = open(loss_history_file, 'w')
    strheader = '/batches_done/ /loss_discriminator/ /loss generator/\n'
    fout.write(strheader)


    num_images = len(dataloader)
    for epoch in range(args.n_epochs):
        progressbar = tqdm(total=num_images,
                           desc='Epochs {}/{}'.format(epoch, args.n_epochs),
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]')
        time_ini = time.time()
        sumrun_lossG = 0.0
        sumrun_lossD = 0.0

        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.flatten(start_dim=1)
            imgs.cuda()

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            # Generate fake image
            latent_z = generate_Gaussian_noise(args.batch_size)
            fake_image = generator(latent_z)
            # prediction on fake image
            predictions_fake = discriminator(fake_image)
            # compute loss
            loss_G = criterion_G(predictions_fake)
            # backprop algorithm step
            loss_G.backward()
            optimizer_G.step()
            # -------------------

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            # Generate fake image
            latent_z = generate_Gaussian_noise(args.batch_size)
            fake_image = generator(latent_z)
            # prediction on real and fake image
            predictions_real = discriminator(imgs)
            predictions_fake = discriminator(fake_image)
            # compute loss
            loss_D = criterion_D(predictions_real, predictions_fake)
            # backprop algorithm step
            loss_D.backward()
            optimizer_D.step()
            # -------------------

            sumrun_lossG += loss_G.item()
            sumrun_lossD += loss_D.item()

            lossG_partial = sumrun_lossG/(i+1)
            lossD_partial = sumrun_lossD/(i+1)
            progressbar.set_postfix(loss='{0:1.5f}/{1:1.5f}'.format(lossD_partial, lossG_partial))
            progressbar.update(1)

            batches_done = epoch * len(dataloader) + i

            if i % args.monitor_interval == 0:
                discriminator_loss = sumrun_lossD/(i+1)
                generator_loss = sumrun_lossG/(i+1)

                time_now = time.time()
                print('\nBatches done {0}. Batch Size = {1}. Discriminator / Generator Loss = {2:1.5}/{3:1.5}. Time compute = {4}'.
                      format(batches_done, args.batch_size, discriminator_loss, generator_loss, (time_now - time_ini)))
                time_ini = time.time()

                # update loss history file
                strdata = '{0} {1:1.5} {2:1.5}\n'.format(batches_done, discriminator_loss, generator_loss)
                fout.write(strdata)

            # Save Images
            # -----------
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated images, e.g.:

                out_images = fake_image[:25].detach().view(-1, 1, IMAGE_DIM, IMAGE_DIM)
                nameoutfile = 'images/{}.png'.format(batches_done)
                save_image(out_images, os.path.join(args.work_dir, nameoutfile), nrow=5, normalize=True)

                # Save generator at intermediate steps
                nameoutfile = 'mnist_generator_batchdone-{}.pt'.format(batches_done)
                torch.save(generator.state_dict(), os.path.join(args.work_dir, nameoutfile))
        #end
    #end

    print('Done training.')
    fout.close()


def create_interpolation_levels_invector(vector_a, vector_b, num_levels=7):
    # interpolation levels in ref. interval [0, 1]
    inter_x_ref = torch.linspace(0,1,num_levels+2) # (+2): include '0' and '1
    output = Variable(torch.zeros(num_levels+2, args.latent_dim))
    for i in range(num_levels+2):
        output[i,:] = vector_a * (1.0 - inter_x_ref[i]) + vector_b * inter_x_ref[i]
    #end
    return output


def predict_interpolate(generator):
    generator.eval()

    # 1st: Generate a batch of noise values in latent space.
    # 2nd: Pick two noise values, and generate 7 interpolated samples in between them
    # 3rd: Generate 9 fake images from these noise samples (2 picked original and 7 interpolated)
    #      (make sure the two picked noise values result in generated fake images from different classes)

    latent_z = generate_Gaussian_noise(2)
    inter_latent_z = create_interpolation_levels_invector(latent_z[0], latent_z[1], num_levels=7)

    inter_fake_images = generator(inter_latent_z)

    out_images = inter_fake_images.detach().view(-1, 1, IMAGE_DIM, IMAGE_DIM)
    nameoutfile = 'images/res_interpolated_images.png'
    save_image(out_images, os.path.join(args.work_dir, nameoutfile), nrow=9, normalize=True)


def main():
    # Create output image directory
    outimgs_dir = os.path.join(args.work_dir,'images/')
    if not os.path.exists(outimgs_dir):
        os.makedirs(outimgs_dir) #exist_ok=True)

    if args.type_experiment=='train':
        # load data
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(CODE_DIR, 'data/mnist/'), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize((0.5, 0.5, 0.5),
                               #                     (0.5, 0.5, 0.5))])
                               transforms.Normalize(mean=[0.5], std=[0.5])])),  # otherwise the code crashes
            batch_size=args.batch_size, shuffle=True)

        # Initialize models and optimizers
        n_features_imgs = IMAGE_DIM * IMAGE_DIM
        generator = Generator(args.latent_dim, n_features_imgs)
        discriminator = Discriminator(n_features_imgs)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

        # Start training
        train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

        # You can save your generator here to re-use it to generate images for your report, e.g.:
        torch.save(generator.state_dict(), os.path.join(args.work_dir,'mnist_generator.pt'))

    else: # args.type_experiment=='predict':
        # Restart model from saved generator
        n_features_imgs = IMAGE_DIM * IMAGE_DIM
        generator = Generator(args.latent_dim, n_features_imgs)
        restartfilename = os.path.join(args.work_dir,'mnist_generator.pt')
        generator.load_state_dict(torch.load(restartfilename, map_location='cuda:0'))

        predict_interpolate(generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_experiment', type=str, default='train',
                        help='Train or Predict?')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--monitor_interval', type=int, default=50,
                        help='monitor losses every MONITOR_INTERVAL iterations')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--work_dir', type=str, default=WORK_DIR_DEFAULT,
                        help='Working directory to store output data')
    args = parser.parse_args()

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    main()
