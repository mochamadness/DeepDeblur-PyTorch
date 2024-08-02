"""main file that does everything"""
import cv2
from utils import interact

from option import args, setup, cleanup
from data import Data
from model import Model
from loss import Loss
from optim import Optimizer
from train import Trainer

def main_worker(rank, args):
    args.rank = rank
    args = setup(args)
    print(args)
    loaders = Data(args).get_loader()
    model = Model(args)
    model.parallelize()
    optimizer = Optimizer(args, model)

    criterion = Loss(args, model=model, optimizer=optimizer)

    trainer = Trainer(args, model, criterion, optimizer, loaders)

    if args.stay:
        interact(local=locals())
        exit()

    if args.demo:
        a = trainer.evaluate(epoch=args.start_epoch, mode='demo')
        exit()

    for epoch in range(1, args.start_epoch):
        if args.do_validate:
            if epoch % args.validate_every == 0:
                trainer.fill_evaluation(epoch, 'val')
        if args.do_test:
            if epoch % args.test_every == 0:
                trainer.fill_evaluation(epoch, 'test')

    for epoch in range(args.start_epoch, args.end_epoch+1):
        if args.do_train:
            trainer.train(epoch)

        if args.do_validate:
            if epoch % args.validate_every == 0:
                if trainer.epoch != epoch:
                    trainer.load(epoch)
                trainer.validate(epoch)

        if args.do_test:
            if epoch % args.test_every == 0:
                if trainer.epoch != epoch:
                    trainer.load(epoch)
                trainer.test(epoch)

        if args.rank == 0 or not args.launched:
            print('')

    trainer.imsaver.join_background()

    cleanup(args)

def deblur_img(path_to_img_dir, rank, args):
    args.demo_input_dir = path_to_img_dir
    args.rank = rank
    args = setup(args)
    loaders = Data(args).get_loader()
    model = Model(args)
    model.parallelize()
    optimizer = Optimizer(args, model)

    criterion = Loss(args, model=model, optimizer=optimizer)

    trainer = Trainer(args, model, criterion, optimizer, loaders)

    if args.demo:
        return (trainer.evaluate(epoch=args.start_epoch, mode='demo'))
    else: print('go away')
def main():
    main_worker(args.rank, args)

if __name__ == "__main__":
    main()