import os.path
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from utils.util import tensor2im, save_image
from torchvision import transforms
from argparse import Namespace
from bisect import bisect
from random import choice
from globals import PART_SIZE_MAGIC, ORIENTATION_MAGIC, HORIZONTAL, VERTICAL, NAME_MAGIC
from puzzle.puzzle_utils import get_full_pair_example_name, get_info_from_file_name, set_orientation_in_name


class VirtualPuzzleDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_false_examples', type=int, default=0,
                            help='What is the ratio of false neighbor to true neighbor examples in the dataset')
        parser.add_argument('--part_size', type=int, default=64,
                            help='What size are the puzzle parts produced by the dataset (The image width and height'
                                 'must be a multiple of "part_size"')
        parser.add_argument('--puzzle_name', type=str, default='',
                            help="Specify a single puzzle name if you want to it to be the only one in the dataset")
        parser.set_defaults(resize_or_crop='none')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.images = []
        self.images_index_dict = {}

        # The folder containing images of true adjacent puzzle pieces according to 'opt.phase' (train / test)
        self.phase_folder = os.path.join(self.root, opt.phase)

        # Paths of the full puzzle images
        self.paths = sorted(make_dataset(self.phase_folder))
        self.transform = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5))])
        self.load_base_images()

    def load_base_images(self):
        num_examples_accumulated = 0
        for path in [p for p in self.paths if PART_SIZE_MAGIC + str(self.opt.part_size) in p and
                                              ORIENTATION_MAGIC + HORIZONTAL in p and
                                              NAME_MAGIC + str(self.opt.puzzle_name) in p]:
            current_image = Namespace()
            current_image.image_dir = os.path.dirname(path)
            current_image.name_horizontal, current_image.image_extension = os.path.splitext(os.path.basename(path))
            current_image.name_vertical = set_orientation_in_name(current_image.name_horizontal, VERTICAL)

            current_image.horizontal = self.get_real_image(os.path.join(
                current_image.image_dir,
                current_image.name_horizontal + current_image.image_extension))

            current_image.vertical = self.get_real_image(os.path.join(
                current_image.image_dir,
                current_image.name_vertical + current_image.image_extension))

            _, height, width = current_image.horizontal.shape
            assert height % self.opt.part_size == 0 and width % self.opt.part_size == 0,\
                "Image wasn't cropped to be a multiple of 'part_size'"
            current_image.num_x_parts = int(width / self.opt.part_size)
            current_image.num_y_parts = int(height / self.opt.part_size)
            current_image.parts_range = range(current_image.num_x_parts * current_image.num_y_parts)
            current_image.num_horizontal_examples = self.count_pair_examples_in_image(current_image.num_x_parts,
                                                                                      current_image.num_y_parts)
            # x and y are reversed in vertical
            current_image.num_vertical_examples = self.count_pair_examples_in_image(current_image.num_y_parts,
                                                                                    current_image.num_x_parts)
            current_image.num_examples = current_image.num_horizontal_examples + current_image.num_vertical_examples
            num_examples_accumulated += current_image.num_examples
            current_image.num_examples_accumulated = num_examples_accumulated
            self.images_index_dict[current_image.name_horizontal] = len(self.images)
            self.images.append(current_image)

    def __getitem__(self, index):
        example = self.get_pair_example_by_index(index)
        example['burnt'] = self.burn_image(example['real'])
        return example

    def __len__(self):
        if len(self.images) == 0:
            return 0
        else:
            return self.images[-1].num_examples_accumulated

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
            example = self.get_pair_example_from_specific_image(image.horizontal, image.num_x_parts,
                                                                image.parts_range, relative_index)
            image_name = image.name_horizontal
        elif relative_index < image.num_examples:
            relative_index -= image.num_horizontal_examples
            # num_y_parts is the number of x parts in the vertical image
            example = self.get_pair_example_from_specific_image(image.vertical, image.num_y_parts,
                                                                image.parts_range, relative_index)
            image_name = image.name_vertical
        else:
            raise IndexError("Invalid index: {0}".format(relative_index))
        example['name'] = get_full_pair_example_name(image_name, example['part1'], example['part2'])
        return example

    def get_pair_example_by_name(self, image_name, part1, part2):
        real_pair = self.crop_pair_by_image_name(image_name, part1, part2)

########## Temporary JPG bug fix ###########################
        #orientation = get_info_from_file_name(image_name, ORIENTATION_MAGIC)

        #temp_folder = os.join('temp', image_name)
        #comparison_folder = os.path.join(r'C:\SHARE\images\pair_inputs_for_completion\test', image_name)
        #temp_file_name = os.path.join(temp_folder, "{0}_{1}_{2}.jpg".format(part1, orientation, part2))
        #comparison_file_name = os.path.join(comparison_folder, "{0}_{1}_{2}.jpg".format(part1 + 1, orientation, part2 + 1))

        #real_pair_numpy = tensor2im(torch.unsqueeze(real_pair, 0))
        #save_image(real_pair_numpy, temp_file_name)
        #real_pair = self.get_real_image(comparison_file_name)
        #comparison_pair = self.get_real_image(comparison_file_name)
        #if (real_pair != comparison_pair).nonzero().shape[0] == 0:
            #os.rename(temp_file_name, os.path.join(temp_folder, "same-"+os.path.basename(temp_file_name)))
##########################################################
        burnt_pair = self.burn_image(real_pair)
        name = get_full_pair_example_name(image_name, part1, part2)
        return {'real': real_pair, 'burnt': burnt_pair, 'name': name}

    def get_pair_example_from_specific_image(self, specific_image, num_x_parts, parts_range, relative_index):

        # The part that corresponds with relative_index
        part_index = int(relative_index / (self.opt.num_false_examples + 1))
        # The part in the original puzzle (after skipping the rightmost column)
        part1 = int(part_index * num_x_parts / (num_x_parts - 1))
        assert part1 % num_x_parts < num_x_parts - 1, "part1 was selected from rightmost column, sum tin wong"
        # Initialize part2 as the true neighbor
        part2 = part1 + 1
        if relative_index % (self.opt.num_false_examples + 1) == 0:
            # A true example
            label = True
        else:
            label = False
            while (part2 == part1 + 1 or part2 == part1):
                # Randomly select a part until it is valid
                part2 = choice(parts_range)
        pair_tensor = self.crop_pair_from_image(specific_image, num_x_parts, part1, part2)
        return {'part1': part1, 'part2': part2, 'label': label, 'real': pair_tensor}

    def crop_pair_by_image_name(self, image_name, part1, part2):
        image = self._get_image_by_name(image_name)
        orientation = get_info_from_file_name(image_name, ORIENTATION_MAGIC)
        if orientation == HORIZONTAL:
            return self.crop_pair_from_image(image.horizontal, image.num_x_parts, part1, part2)
        elif orientation == VERTICAL:
            return self.crop_pair_from_image(image.vertical, image.num_y_parts, part1, part2)
        else:
            # Invalid
            raise KeyError("No image with valid orientation magic exists for path %s" % image_name)

    def crop_pair_from_image(self, image, num_x_parts, part1, part2):

        part1_tensor = self.crop_part_from_image(image, num_x_parts, part1)
        part2_tensor = self.crop_part_from_image(image, num_x_parts, part2)
        return torch.cat((part1_tensor, part2_tensor), 2)


    def crop_part_from_image(self, image, num_columns, part):
        row = int(part / num_columns)
        column = part % num_columns
        return image[:, row * self.opt.part_size: (row + 1) * self.opt.part_size,
                     column * self.opt.part_size: (column + 1) * self.opt.part_size]

    def get_image_metadata(self, image_name):
        image = self._get_image_by_name(image_name)
        metadata = Namespace()
        metadata.orientation = get_info_from_file_name(image_name, ORIENTATION_MAGIC)
        assert metadata.orientation in [HORIZONTAL, VERTICAL], "Invalid orientation in image_name"
        metadata.full_puzzle_name = image_name
        if metadata.orientation == HORIZONTAL:
            metadata.num_x_parts = image.num_x_parts
            metadata.num_y_parts = image.num_y_parts
        elif metadata.orientation == VERTICAL:
            metadata.num_x_parts = image.num_y_parts
            metadata.num_y_parts = image.num_x_parts
        return metadata

    def _get_image_by_name(self, path):
        horizontal_path = set_orientation_in_name(path, HORIZONTAL)
        image_index = self.images_index_dict[horizontal_path]
        return self.images[image_index]

    def save_pair(self, image_name, part1, part2):
        real_pair_numpy = self.get_pair_numpy(image_name, part1, part2)
        save_image(real_pair_numpy, get_full_pair_example_name(image_name, part1, part2) + ".jpg")

    def get_pair_numpy(self, image_name, part1, part2):
        example = self.get_pair_example_by_name(image_name, part1, part2)
        real_pair = example['real']
        return tensor2im(torch.unzsqueeze(real_pair, 0))






