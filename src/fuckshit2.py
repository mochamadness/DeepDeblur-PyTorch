import cv2
import numpy as np
import torch
from skimage.transform import pyramid_gaussian
import torchvision.transforms as transforms
import imageio

def pad(img, divisor=4, pad_width=None, negative=False):
    def _pad_numpy(img, divisor=4, pad_width=None, negative=False):
        if pad_width is None:
            (h, w, _) = img.shape
            pad_h = -h % divisor
            pad_w = -w % divisor
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))

        img = np.pad(img, pad_width, mode='edge')
        return img, pad_width

    def _pad_tensor(img, divisor=4, pad_width=None, negative=False):
        n, h, w, c = img.shape
        if pad_width is None:
            pad_h = -h % divisor
            pad_w = -w % divisor
            pad_width = (0, pad_w, 0, pad_h)
        else:
            try:
                pad_h = pad_width[0][1]
                pad_w = pad_width[1][1]
                if isinstance(pad_h, torch.Tensor):
                    pad_h = pad_h.item()
                if isinstance(pad_w, torch.Tensor):
                    pad_w = pad_w.item()

                pad_width = (0, pad_w, 0, pad_h)
            except:
                pass

            if negative:
                pad_width = [-val for val in pad_width]

        img = torch.nn.functional.pad(img, pad_width, 'reflect')
        return img, pad_width

    if isinstance(img, np.ndarray):
        return _pad_numpy(img, divisor, pad_width, negative)
    else:  # torch.Tensor
        return _pad_tensor(img, divisor, pad_width, negative)

def generate_pyramid(image, n_scales):
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    pyramid = list(pyramid_gaussian(image, max_layer=n_scales-1, channel_axis=-1))  # multichannel=True is removed from skimg 0.23
    return pyramid

def np2tensor(*args):
    def _np2tensor(x):
        np_transpose = np.ascontiguousarray(x.transpose(2, 0, 1))
        tensor = torch.from_numpy(np_transpose)
        return tensor

    return [_np2tensor(x) for x in args]

def create_input(image_path, n_scales=3):
    # Read the image using cv2
    #image = cv2.imread(image_path)
    # Convert the image to RGB
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = imageio.imread(image_path, pilmode='RGB')

    # Pad the image
    imgs = image

    imgs, pad_width = pad(imgs, divisor=2**(n_scales-1))
    #padded_image, pad_width = pad(image, divisor=2**(n_scales-1))
    # Generate the Gaussian pyramid
    imgs = generate_pyramid(imgs, n_scales)
    # Convert pyramid images to tensors
    imgs = np2tensor(*imgs)
    blur = imgs
    return blur, pad_width
if __name__ == "__main__":
    image_path = '004001.png'  # replace with your image path
    blur, pad_width = create_input(image_path)
    print("Padded image tensor shape:", blur[0].shape)
    print("Pad width:", pad_width)