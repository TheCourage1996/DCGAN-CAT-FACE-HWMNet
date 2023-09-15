import torch
import config
import torchvision
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from losses_HWMNet import CharbonnierLoss
from data_RGB import get_training_data, get_validation_data

writer = SummaryWriter()

def train_fn(dataloader_target, gen , disc , opt_gen , opt_disc , fixed_noise, D_loss,G_loss):
    
    for epoch in range(1, config.NUM_EPOCHS + 1):

        with tqdm(dataloader_target) as progress_bar:
            for real, _ in progress_bar:
                    
                    gen.train()
                    disc.train()
                    
                    real = real.to(config.DEVICE)
                    batch_size = real.shape[0]

                    noise = torch.randn(batch_size , config.Z_DIM , 1 , 1).to(config.DEVICE)
                    input = torch.ones(batch_size, 3, 64, 64, dtype=torch.float, requires_grad=True).cuda()
                    # Training Discriminator
                    fake = gen(input)
                    disc_real = disc(real).reshape(-1)
                    loss_disc_real = D_loss(disc_real, torch.ones_like(disc_real))
                    disc_fake = disc(fake.detach()).reshape(-1)
                    loss_disc_fake = D_loss(disc_fake, torch.zeros_like(disc_fake))
                    loss_disc = (loss_disc_real + loss_disc_fake) / 2

                    #print(loss_disc)
                    opt_disc.zero_grad()
                    loss_disc.backward(retain_graph=True)
                    opt_disc.step()

                    # Train generator

                    output = disc(fake).reshape(-1)
                    loss_gen = G_loss(output, torch.ones_like(disc_real))

                    #print(loss_gen)

                    opt_gen.zero_grad()
                    loss_gen.backward()
                    opt_gen.step()

                    # Display messege
                    progress_bar.set_description(f' - Epoch {epoch}/{config.NUM_EPOCHS} - ')
                    progress_bar.set_postfix(loss_G = loss_gen.item(), loss_D = loss_disc.item())
                    
        # Write the messege to Tensorboard
        writer.add_scalar("D_Loss/Losses", loss_disc , epoch)
        writer.add_scalar("G_Loss/Losses", loss_gen , epoch)
        
        if epoch % config.SAVE_IMG_FREQ == 0:
            with torch.no_grad():

                fake = gen(fixed_noise)
                # img_grid_real = torchvision.utils.make_grid(real[:32], normalize = True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize = True)
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
    ## DataLoaders
    # print('==> Loading datasets')
    # train_dataset = get_training_data(config.ROOT_DIR, {'patch_size': 256})
    # train_loader = DataLoader(dataset=train_dataset, batch_size=2,
    #                           shuffle=True, num_workers=0, drop_last=False)
    # val_dataset = get_validation_data(config.ROOT_DIR, {'patch_size': 64})
    # val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
    #                         drop_last=False)

    # for batch_data, batch_labels in dataloader:
    #     # 将数据传输到 GPU
    #     batch_data = batch_data.to(config.DEVICE)
    #     batch_labels = batch_labels.to(config.DEVICE)

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

    fixed_noise = torch.randn(32 , config.Z_DIM , 1 , 1).to(config.DEVICE)
    input_fixed_noise = torch.ones(32, 3, 400, 592, dtype=torch.float, requires_grad=False).cuda()
    train_fn(dataloader_target, generator, discriminator, opt_gen, opt_disc, fixed_noise, D_loss,G_loss)

        
if __name__  == "__main__":
    main()
