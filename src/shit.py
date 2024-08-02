"""main file that does everything"""
from utils import interact

from option import args, setup, cleanup
from data import Data
from model import Model
from loss import Loss
from optim import Optimizer
from train import Trainer
import cv2

from PIL import Image
from torchvision import transforms

def main_worker(rank, args):
    args.rank = rank
    args = setup(args)

    #loaders = Data(args).get_loader()
    model = Model(args)
    model.parallelize()
    optimizer = Optimizer(args, model)

    criterion = Loss(args, model=model, optimizer=optimizer)

    trainer = Trainer(args, model, criterion, optimizer, 0)
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
    ])
    image_tensor = transform(image)
    deblurred_image =  trainer.deblur_image(epoch=args.start_epoch, image=image_tensor)

    def save_image(tensor, file_path):
        """
        Save a torch.Tensor as an image file.

        Args:
            tensor (torch.Tensor): The image tensor to save.
            file_path (str): The path where the image will be saved.
        """
        # Make sure the tensor is on the CPU and remove the batch dimension if present
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.dim() == 4:  # If batch dimension is present
            tensor = tensor.squeeze(0)

        # Convert tensor to PIL Image
        transform = transforms.ToPILImage()
        image = transform(tensor)

        # Save image
        image.save(file_path)

    # Example usage:
    # Assuming `deblurred_image` is the tensor obtained from the deblur_image function
    file_path = 'deblurred_image.jpg'
    save_image(deblurred_image, file_path)


def main():
    main_worker(args.rank, args)

if __name__ == "__main__":
    main()