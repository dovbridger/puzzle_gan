import os.path
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from torchvision import transforms
from argparse import Namespace
from bisect import bisect


class VirtualPuzzleDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_false_examples', type=int, default=0,
                            help='What is the ratio of false neighbor to true neighbor examples in the dataset')
        parser.add_argument('--part_size', type=int, default=64,
                            help='What size are the puzzle parts produced by the dataset (The image width and height'
                                 'must be a multiple of "part_size"')
        parser.set_defaults(resize_or_crop='none')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.images = []

        # The folder containing images of true adjacent puzzle pieces according to 'opt.phase' (train / test)
        self.phase_folder = os.path.join(self.root, opt.phase)

        # Paths of the full puzzle images
        self.paths = sorted(make_dataset(self.phase_folder))
        self.transform = get_transform(opt)
        self.load_base_images()

    def load_base_images(self):
        num_examples_accumulated = 0
        for path in self.paths:
            horizontal_image = self.get_real_image(self, path)
            vertical_image = transforms.functional.rotate(horizontal_image, 90)
            current_image = Namespace()
            current_image.path = path
            current_image.horizontal = horizontal_image
            current_image.vertical = vertical_image
            _, current_image.num_y_parts, current_image.num_x_parts = horizontal_image.shape
            current_image.num_horizontal_examples = self.count_pair_examples_in_image(current_image.num_x_parts,
                                                                                      current_image.num_y_parts)
            # x and y are reversed in vertical
            current_image.num_vertical_examples = self.count_pair_examples_in_image(current_image.num_y_parts,
                                                                                    current_image.num_x_parts)
            current_image.num_examples = current_image.num_horizontal_examples + current_image.num_vertical_examples
            num_examples_accumulated += current_image.num_examples
            current_image.num_examples_accumulated = num_examples_accumulated
            self.images.append(current_image)

    def __getitem__(self, index):
        example = self.get_pair_example_by_index(index)
        example['burnt_pair_image'] = self.burn_image(example['real_pair_image'])
        return example

    def __len__(self):
        return len(self.paths)

    def get_real_image(self, path):
        original_img = Image.open(path).convert('RGB')
        # Perform the transformations
        real_image = self.transform(original_img)

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
        return 'VirtualPuzzleDataset'

    def determine_label(self):
        pass

    def count_pair_examples_in_image(self, num_x_parts, num_y_parts):
        num_true_examples = num_y_parts * (num_x_parts - 1)
        return num_true_examples * (self.opt.num_false_examples + 1)

    def get_pair_example_by_index(self, index):
        relevant_image_index = bisect([x.num_examples_accumulated for x in self.images], index)
        relevant_image = self.images[relevant_image_index]
        num_previous_examples = relevant_image.num_examples_accumulated - relevant_image.num_examples
        return self.get_pair_example_by_relative_index(relevant_image, index - num_previous_examples)

    def get_pair_example_by_relative_index(self, image, relative_index):
        if relative_index < image.num_horizontal_examples:
           example = self.get_pair_example_from_specific_image(image.horizontal, image.num_x_parts, relative_index)
        elif relative_index < image.num_vertical_examples:
            relative_index -= image.num_horizontal_examples
            # num_y_parts is the number of x parts in the vertical image
            example = self.get_pair_example_from_specific_image(image.vertical, image.num_y_parts, relative_index)
        else:
            raise IndexError("Invalid index: {0}".format(relative_index))
        example['path'] = image.path
        return example

    def get_pair_example_from_specific_image(self, specific_image, num_x_parts, relative_index):
        part1 = int(relative_index / (num_x_parts - 1))
        if relative_index % (self.opt.num_false_examples + 1) == 0:
            # A true example
            label = True
            part2 = part1 + 1
        else:
            label = False
            # Temporary
            part2 = 0
        pair_tensor = self.crop_pair_from_image(specific_image, num_x_parts, part1, part2)
        return {'part1': part1, 'part2': part2, 'label': label, 'real_pair_image': pair_tensor}

    def crop_pair_from_image(self, image, num_x_parts, part1, part2):

        part1_tensor = self.crop_part_from_image(image, num_x_parts, part1)
        part2_tensor = self.crop_part_from_image(image, num_x_parts, part2)
        return torch.cat((part1_tensor, part2_tensor), 2)


    def crop_part_from_image(self, image, num_columns, part):
        row = int(part / num_columns)
        column = part % num_columns
        return image[:, row * self.opt.part_size: (row + 1) * self.opt.part_size,
                     column * self.opt.part_size: (column + 1) * self.opt.part_size]



