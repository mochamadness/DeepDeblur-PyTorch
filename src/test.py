import imageio
import numpy as np
import main
from utils import interact
import cv2
from option import args, setup, cleanup
from data import Data
from model import Model
from loss import Loss
from optim import Optimizer
from train import Trainer
import argparse
import datetime
import os
import re
import shutil
import time

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from utils import interact
from utils import str2bool, int2str

import template


now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# Load your image
args.save_dir = r'D:\Projects\DeepDeblur-PyTorch\experiment\GOPRO_L1' #get model location
path_to_img_dir = r'D:\Projects\DeepDeblur-PyTorch\src\shit'
args.demo_input_dir = path_to_img_dir
args.demo = True
args.demo_output_dir = r'D:\Projects\DeepDeblur-PyTorch\src'
if args.demo:
    assert os.path.basename(args.save_dir) != now, 'You should specify pretrained directory by setting --save_dir SAVE_DIR'

    args.data_train = ''
    args.data_val = ''
    args.data_test = ''

    args.do_train = False
    args.do_validate = False
    args.do_test = False

    assert len(args.demo_input_dir) > 0, 'Please specify demo_input_dir!'
    args.demo_input_dir = os.path.expanduser(args.demo_input_dir)
    if args.demo_output_dir:
        args.demo_output_dir = os.path.expanduser(args.demo_output_dir)

    args.save_results = 'all'
args.start_epoch = 1001
args.load_epoch = 1000
print(args)
output_tensor = main.main_worker(args.rank, args)
#print(output_tensor)
