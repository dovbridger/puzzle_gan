import os.path
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class PuzzleDataset(BaseDataset):
    '''
    Dov's implementation of a dataset to be used for the puzzle problem
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # How many pixels will be missing from each edge of a puzzle part
        # (The hole between two parts will be 2 * burn_extent)
        parser.add_argument('--burn_extent', type=int, default=2,
                            help='Number of pixel columns missing on the edge of each puzzle piece')

        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # The folder containing images of true adjacent puzzle pieces according to 'opt.phase' (train / test)
        self.phase_folder = os.path.join(self.root, opt.phase)

        # Paths of the images of true adjacent puzzle pieces
        self.paths = sorted(make_dataset(self.phase_folder))

        # Transformations that need to be done on input images before feeding them to the network
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        '''
        get a single example from from the dataset
        :param index: index of them image you want (images are sorted according to path)
        :return: dict containing the real image (ground truth), the burnt image (real minus the pixels that are burnt)
                 and the path.
        '''

        path = self.paths[index]
        real_image = self.get_real_image(path)
        burnt_image = self.burn_image(real_image)

        return {'real': real_image, 'burnt': burnt_image, 'path': path}

    def __len__(self):
        return len(self.paths)

    def get_real_image(self, path):
        original_img = Image.open(path).convert('RGB')
        # Perform the transformations
        real_image = self.transform(original_img)

        # If number of input channels requested is 1, convert RGB to gray
        if self.opt.input_nc == 1:
            tmp = real_image[0, ...] * 0.299 + \
                  real_image[1, ...] * 0.587 + \
                  real_image[2, ...] * 0.114
            real_image = tmp.unsqueeze(0)
        return real_image

    def burn_image(self, real_image):
        '''
        Sets a (2 * opt.burn_extent) column of pixels to -1 in the center of a cloned instance of real_image
        :param real_image: The input image
        :return: The burnt image
        '''
        burnt_image = torch.clone(real_image)
        channels, height, width = burnt_image.shape
        center = int(width / 2)
        burn_extent = self.opt.burn_extent
        burnt_image[:, :, center - burn_extent: center + burn_extent] = -1
        return burnt_image

    def name(self):
        return 'PuzzleDataset'
