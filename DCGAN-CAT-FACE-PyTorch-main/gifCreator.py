import os
import imageio
from tqdm import tqdm
from argparse import ArgumentParser

def gifCreator(dir, output_dir = 'out.gif', fps = 1):
    print('=' * 60)
    print(' - Processing starts -')
    imgs = []
    loader = tqdm(os.listdir(dir))

    counter = 1
    for filename in loader:
        loader.set_description(' - GIF Creator ')
        imgs.append(imageio.imread(dir + 'fake_ep' + str(counter) + '.png'))
        counter += 1

    imageio.mimsave(output_dir, imgs , fps = fps)
    print(' - Processing successfully finished. GIF saved to ', output_dir, ' - ')
    print('=' * 60)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--DIR', help = 'direction to images', type = str, default='saved_samples/')
    parser.add_argument('--OUTPUT_DIR', help = 'output direction', type = str, default='output.gif')
    parser.add_argument('--FPS', help = 'determine the fps value', type=float, default=15)     # less -> slow
    args = parser.parse_args()
    
    gifCreator(dir = args.DIR, output_dir = args.OUTPUT_DIR, fps = args.FPS)