import torch
import config
import torchvision
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from losses_HWMNet import CharbonnierLoss
import numpy as np
import matplotlib.pyplot as plt

writer = SummaryWriter()


def train_fn(dataloader_target, gen , disc , opt_gen , opt_disc , fixed_noise, D_loss,G_loss):
    D=[]
    G=[]
    for epoch in range(1, config.NUM_EPOCHS + 1):
        d_epoch_loss=0
        g_epoch_loss=0
        count=len(dataloader_target)

        for step,(real,_) in enumerate(dataloader_target):
            # ########################

            #        训练判别器       #

            # ########################
            real = real.to(config.DEVICE) # 真实图片
            batch_size = real.shape[0]
            ##添加随机噪音
            #noise = torch.randn(batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
            input_noise = torch.ones(batch_size, 3, 64, 64, dtype=torch.float).cuda()
            gen_imag = gen(input_noise)  # 从噪声中生成假数据
            real_out = disc(real)
            fake_out = disc(gen_imag)
            opt_disc.zero_grad()

            d_real_loss = D_loss(real_out, torch.ones_like(real_out))
            d_fake_loss = D_loss(fake_out, torch.zeros_like(fake_out))
            d_loss = (d_fake_loss + d_real_loss)/2
            d_loss.backward(retain_graph=True)
            opt_disc.step()

            # ########################

            #        训练生成器       #

            # ########################
            g_loss = G_loss(fake_out, torch.ones_like(fake_out))
            opt_gen.zero_grad()
            g_loss.backward()
            opt_gen.step()

            with torch.no_grad():
                d_epoch_loss += d_loss
                g_epoch_loss += g_loss

        with torch.no_grad():
            d_epoch_loss/=count
            g_epoch_loss/=count
            D.append(d_epoch_loss)
            G.append(g_epoch_loss)
            print('epoch:',epoch)
            print('d_epoch_loss:',d_epoch_loss)
            print('g_epoch_loss:', g_epoch_loss)

        
        if epoch % config.SAVE_IMG_FREQ == 0:
            with torch.no_grad():

                fake = gen(fixed_noise)
                # img_grid_real = torchvision.utils.make_grid(real[:32], normalize = True)
                img_grid_fake = torchvision.utils.make_grid(fake[:2], normalize = True)
                plot_examples( "fake_ep" + str(epoch) + ".png" , "saved_samples/" , img_grid_fake)
                # plot_examples("Real_ep"+str(epoch)+ ".png" , "saved_samples/" , img_grid_real)
        
        if epoch % config.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(gen , opt_gen , filename = 'ep' + str(epoch) + '_' + config.CHECKPOINT_GEN, dir = config.CHECKPOINT_DIR)
            save_checkpoint(disc , opt_disc , filename = 'ep' + str(epoch) + '_' + config.CHECKPOINT_DISC, dir = config.CHECKPOINT_DIR)
            messegeDividingLine(f' - Checkpoint saved! - ')
    
def main():
    checkGPU(config)
    transforms = getTransform(config)

    dataloader_target = getDataLoader(config, transforms, config.ROOT_DIR)

    generator, discriminator = getModel(config)
    generator.to(config.DEVICE)
    discriminator.to(config.DEVICE)
    opt_gen, opt_disc = getOptimizer(config, generator.parameters(), discriminator.parameters())
    D_loss = getLossFunction().to(config.DEVICE)
    G_loss = CharbonnierLoss().to(config.DEVICE)

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, generator, opt_gen, config.G_LEARNING_RATE).to(config.DEVICE)
        load_checkpoint(config.CHECKPOINT_DISC , discriminator , opt_disc , config.D_LEARNING_RATE).to(config.DEVICE)

    messegeDividingLine(" - Training starts -")

    fixed_noise = torch.randn(32 , config.DEPTH , 1 , 1).to(config.DEVICE)
    input_fixed_noise = torch.ones(2, 3, 64, 64, dtype=torch.float, requires_grad=False).cuda()
    train_fn(dataloader_target, generator, discriminator, opt_gen, opt_disc, input_fixed_noise, D_loss,G_loss)

        
if __name__  == "__main__":
    main()
