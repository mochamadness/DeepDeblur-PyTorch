"""main file that does everything"""
from utils import interact

from option import args, setup, cleanup
from data import Data
from model import Model
from loss import Loss
from optim import Optimizer
from train import Trainer
import cv2
import fuckshit2
from PIL import Image
from torchvision import transforms

def main_worker(rank, args):
    image_path = '004001.png'
    blur, pad_width = fuckshit2.create_input(image_path)
    args.rank = rank
    args = setup(args)

    loaders = Data(args).get_loader()
    model = Model(args)
    model.parallelize()
    optimizer = Optimizer(args, model)

    criterion = Loss(args, model=model, optimizer=optimizer)

    trainer = Trainer(args, model, criterion, optimizer, loaders)

    if args.stay:
        interact(local=locals())
        exit()

    """ if args.demo:
        print('aint no fucking around')
        trainer.cunts(epoch=args.start_epoch, input_img=blur, pad_width=pad_width)
        exit() """

    if args.demo:
        trainer.fuckingaround(epoch=args.start_epoch, mode='demo')
        exit()   



def main():
    main_worker(args.rank, args)

if __name__ == "__main__":
    main()