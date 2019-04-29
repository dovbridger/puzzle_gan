from __future__ import print_function
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from globals import DATASET_MEAN, DATASET_STD


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8, opt=None):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    mean, std = get_dataset_mean_std(opt)
    num_channels = image_tensor.shape[1]
    mean = mean[0:num_channels]
    std = std[0:num_channels]
    inverse_mean = [-m / s for m, s in zip(mean, std)]
    inverse_std = [1/s for s in std]
    inverse_transform = transforms.Normalize(mean=inverse_mean, std=inverse_std)
    image_temp = image_tensor[0].cpu().float()
    image_numpy = inverse_transform(image_temp).numpy() * 255.0
    if num_channels == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    #image_numpy = (np.transpose(image_temp, (1, 2, 0)) + 1) / 2.0 * 255.0

    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def get_dataset_mean_std(opt):
    if opt is None or not opt.use_specific_normalize:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        return DATASET_MEAN, DATASET_STD


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



