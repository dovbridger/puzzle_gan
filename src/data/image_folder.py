###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    '''
    collect image paths under the root directory 'dir'
    :param dir: path to the root directory of the images of the dataset that we want to make
    :return: List of all image paths
    '''
    image_paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, file_names in sorted(os.walk(dir)):
        for file_name in file_names:
            if is_image_file(file_name):
                path = os.path.join(root, file_name)
                image_paths.append(path)

    return image_paths


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    '''
    Class that represents a folder with images.
    The images are saved on disk in their original form but can be retrieved in different transformations according
    what is specified in 'transform'
    '''
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        '''

        :param index: index of the image that we want
        :return: A transformed version of the image
        '''
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
