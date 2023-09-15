import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from HWMNet import HWMNet
from networks_ver4 import define_D

def mkdir(direction):
    if not os.path.exists(direction):
        os.makedirs(direction)

def getModel(config):
    from model import Discriminator
    #print('加载生成器模型')
    gen = HWMNet(config.DEPTH, config.CHANNELS_IMG, config.FEATURES_GEN).to(config.DEVICE)
    #print('加载判别器模型')
    disc  = define_D(config.CHANNELS_IMG , config.FEATURES_CRITIC).to(config.DEVICE)
    print('模型加载完毕')
    # if config.SHOW_MODEL:
    #     print(gen)
    #     print('=' * 40)
    #     print(disc)
    return gen, disc

def getDataLoader( config, transforms, dir):
    dataset = datasets.ImageFolder(root=dir, transform=transforms)

    loader = DataLoader(
        dataset,
        batch_size = config.BATCH_SIZE,
        shuffle=True,
    )

    return loader

def getOptimizer(config, gen_params, disc_params):
    opt_gen = optim.Adam(gen_params , lr = config.G_LEARNING_RATE , betas = (0.5 , 0.999))
    opt_disc = optim.Adam(disc_params , lr = config.D_LEARNING_RATE , betas = (0.5 , 0.999))
    return opt_gen, opt_disc

def getLossFunction():
    return nn.BCELoss()

def checkGPU(config):
    if config.DEVICE == 'cuda':
        if torch.cuda.is_available():
            messegeDividingLine('Using GPU for training! (つ´ω`)つ')
            return 'cuda'
        else:
            messegeDividingLine('No GPU! 。･ﾟ･(つд`)･ﾟ･')
            return 'cpu'
    elif config.DEVICE == 'cpu':
        if torch.cuda.is_available():
            messegeDividingLine('Using cpu for training! (つ´ω`)つ')
    else:
        raise KeyError(f'Wrong indication for the device: {config.DEVICE}')

def messegeDividingLine(messege):
    print('=' * 40)
    print(messege)
    print('=' * 40)

def getTransform(config):
    return transforms.Compose([
                                transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5 for _ in range(config.CHANNELS_IMG)], [0.5 for _ in range(config.CHANNELS_IMG)]),
                              ])

def save_checkpoint(model , optimizer , filename = "my_checkpoint.pth.tar", dir = 'checkpoints/'):
    mkdir(dir)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint , dir + filename)
    

def load_checkpoint(checkpoint_file , model , optimizer , lr):
    print("=> Loading checkpoint...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_file , map_location = DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def plot_examples(filename , folder , img):
    mkdir(f"{folder}/")
    save_image(img  , f"{folder}/{filename}")
    
