import os.path
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class PuzzleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--burn_extent', type=int, default=3,
                            help='Number of pixel columns missing on the edge of each puzzle piece')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase_folder_true = os.path.join(self.root, opt.phase, 'True')
        self.phase_folder_false = os.path.join(self.root, opt.phase, 'False')
        self.true_paths = sorted(make_dataset(self.phase_folder_true))
        self.false_paths = sorted(make_dataset(self.phase_folder_false))
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = self.true_paths[index]
        real_image = self.get_real_image(path)
        burnt_image = self.burn_image(real_image)

        return {'real': real_image, 'burnt': burnt_image, 'path': path}

    def __len__(self):
        return len(self.true_paths)

    def get_real_image(self, path):
        original_img = Image.open(path).convert('RGB')
        real_image = self.transform(original_img)

        if self.opt.input_nc == 1:  # RGB to gray
            tmp = real_image[0, ...] * 0.299 + \
                  real_image[1, ...] * 0.587 + \
                  real_image[2, ...] * 0.114
            real_image = tmp.unsqueeze(0)
        return real_image

    def burn_image(self, real_image):
        burnt_image = torch.clone(real_image)
        channels, height, width = burnt_image.shape
        center = int(width / 2)
        burn_extent = self.opt.burn_extent
        burnt_image[:, :, center - burn_extent: center + burn_extent] = -1
        return burnt_image

    def get_false_paths(self, true_path):
        file_name_true_without_extension = os.path.basename(true_path).split('.')[0]
        return [file for file in self.false_paths if os.path.basename(file).startswith(file_name_true_without_extension)]

    def name(self):
        return 'PuzzleDataset'
