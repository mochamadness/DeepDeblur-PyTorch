import imageio
import numpy as np
import torch
from skimage.transform import pyramid_gaussian
import cv2
from data import common
from option import args, setup, cleanup

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
        n, c, h, w = img.shape
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
    pyramid = list(pyramid_gaussian(image, max_layer=n_scales-1, channel_axis=-1)) #multichannel=True is removed from skimg 0.23
    return pyramid

def np2tensor(*args):
    def _np2tensor(x):
        np_transpose = np.ascontiguousarray(x.transpose(2, 0, 1))
        tensor = torch.from_numpy(np_transpose)
        return tensor

    return [_np2tensor(x) for x in args]

def get_image(image_path):
    #image = imageio.imread(image_path, pilmode='RGB')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def create_pyramid_tensors(image, n_scales=3):
    # Pad the image
    imgs = [image]
    imgs = common.np2tensor(*imgs)
    print(imgs[0].shape)
    imgs[0] = imgs[0].unsqueeze(0)
    print(imgs[0].shape)
    pad_width = 0   # dummy value
    imgs[0], pad_width = common.pad(imgs[0], divisor=2**(args.n_scales-1))
    imgs = common.generate_pyramid(*imgs, n_scales=args.n_scales)
    imgs = common.np2tensor(*imgs)
    blur = imgs[0]
    sharp = imgs[1] if len(imgs) > 1 else False
    return blur, sharp, pad_width

if __name__ == "__main__":
    image_path = '004001.png'  # replace with your image path
    image = get_image(image_path)
    batch = create_pyramid_tensors(image)
    print("Gaussian pyramid created. First tensor shape:", batch[0])
    print('Pad width: ', batch[2])
    dtype_eval = torch.float32 if args.precision == 'single' else torch.float16
    input, target = common.to(
        batch[0], batch[1], device=torch.device(args.device_type, args.device_index) if args.device_type == 'cuda' else torch.device(args.device_type), dtype=dtype_eval)
    print(input[0].shape)
