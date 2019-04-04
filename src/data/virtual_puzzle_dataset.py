import os.path
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.virtual_image import VirtualImage

from PIL import Image
from utils.util import tensor2im, save_image, mkdir
from argparse import Namespace
from bisect import bisect
from random import choice
from globals import ORIENTATION_MAGIC, HORIZONTAL, VERTICAL, NAME_MAGIC, METADATA_FILE_NAME, METADATA_FOLDER_NAME,\
    METADATA_DELIMITER, DELIMITER_MAGIC
from puzzle.puzzle_utils import get_full_pair_example_name, get_info_from_file_name, set_orientation_in_name,\
    get_full_puzzle_name_from_characteristics
from puzzle.java_utils import get_java_diff_file, parse_3d_numpy_array_from_json, get_top_k_neighbors,\
    convert_orientation_to_index

MAX_NEIGHBOR_LIMIT = 9999
SAVE_CROPPED_IMAGES = False
CROPPED_IMAGES_FOLDER = 'cropped'

class VirtualPuzzleDataset(BaseDataset):
    transform = None

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_false_examples', type=int, default=0,
                            help='What is the ratio of false neighbor to true neighbor examples in the dataset')
        parser.add_argument('--part_size', type=int, default=64,
                            help='What size are the puzzle parts produced by the dataset (The image width and height'
                                 'must be a multiple of "part_size"')
        parser.add_argument('--puzzle_name', type=str, default='',
                            help="Specify a single puzzle name if you want to it to be the only one in the dataset")
        parser.add_argument('--max_neighbor_rank', type=int, default=-1,
                            help='Only use false neighbors that are ranked up to this number in the original'
                                 '0-burn diff matrix. Min value = 2. 1 or less means use all ranks')
        parser.set_defaults(resize_or_crop='crop_to_part_size')
        parser.set_defaults(nThreads=0)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.images = []
        self.images_index_dict = {}

        self.phase_folder = os.path.join(self.root, opt.phase)

        # Paths of the full puzzle images
        self.paths = sorted(make_dataset(self.phase_folder))
        VirtualPuzzleDataset.transform = get_transform(opt)
        print("Loading base images")
        self.load_base_images()
        print("Base images are loaded")
        self.set_neighbor_choice_options_limit(opt.max_neighbor_rank if opt.max_neighbor_rank >=2 else MAX_NEIGHBOR_LIMIT)
        if opt.phase == 'test':
            try:
                self.create_base_images_metadata()
            except Exception as e:
                print("problem creating metadata, exception:{0}".format(str(e)))

    def load_base_images(self):
        num_examples_accumulated = 0
        for path in [p for p in self.paths if ORIENTATION_MAGIC + HORIZONTAL in p and
                     NAME_MAGIC + str(self.opt.puzzle_name) in p]:
            current_image = VirtualImage(path, self.opt, num_examples_accumulated)
            num_examples_accumulated = current_image.num_examples_accumulated
            if SAVE_CROPPED_IMAGES:
                self.save_cropped_image(current_image)
            # Delete the raw images and replace them with individual parts arrays
            current_image.create_individaul_parts()
            self.images_index_dict[current_image.name_horizontal] = len(self.images)
            self.images.append(current_image)

    def save_cropped_image(self, image):
        cropped_folder = os.path.join(self.phase_folder, CROPPED_IMAGES_FOLDER)
        mkdir(cropped_folder)
        image_numpy = tensor2im(image.horizontal.unsqueeze(0))
        puzzle_name = get_info_from_file_name(image.name_horizontal, NAME_MAGIC)
        image_path = os.path.join(cropped_folder, puzzle_name + image.image_extension)
        save_image(image_numpy, image_path)

    def create_base_images_metadata(self):
        metadata_folder = os.path.join(self.phase_folder, METADATA_FOLDER_NAME)
        mkdir(metadata_folder)
        for image in self.images:
            metadata = self.get_image_metadata(image.name_horizontal)
            puzzle_name = get_info_from_file_name(image.name_horizontal, NAME_MAGIC)
            file_name = get_full_puzzle_name_from_characteristics(puzzle_name=puzzle_name,
                                                                  part_size=self.opt.part_size,
                                                                  orientation=HORIZONTAL) +\
                        DELIMITER_MAGIC + METADATA_FILE_NAME
            self.write_metadata_file(metadata, os.path.join(metadata_folder, file_name))

    @staticmethod
    def write_metadata_file(metadata, file_path):
        metadata_str = METADATA_DELIMITER.join([
            'num_y_parts:' + str(metadata.num_y_parts),
            'num_x_parts:' + str(metadata.num_x_parts),
            'orientation:' + str(metadata.orientation),
        ])
        with open(file_path, 'w') as f:
            f.write(metadata_str)

    def set_neighbor_choice_options_limit(self, num_neighbors):
        if num_neighbors < 2:
            print("WARNING: You must use a minimum of 2 neighbor choices, limit will be 2")
            self.neighbor_choices_limit = 2
        else:
            self.neighbor_choices_limit = num_neighbors


    def __getitem__(self, index):
        example = self.get_pair_example_by_index(index)
        example['burnt'] = self.burn_image(example['real'])
        return example

    def __len__(self):
        if len(self.images) == 0:
            return 0
        else:
            return self.images[-1].num_examples_accumulated

    @staticmethod
    def get_real_image(path):
        original_img = Image.open(path).convert('RGB')
        # Perform the transformations
        real_image = VirtualPuzzleDataset.transform(original_img)

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

    def get_pair_example_by_index(self, index):
        relevant_image_index = bisect([x.num_examples_accumulated for x in self.images], index)
        relevant_image = self.images[relevant_image_index]
        num_previous_examples = relevant_image.num_examples_accumulated - relevant_image.num_examples
        return self.get_pair_example_by_relative_index(relevant_image, index - num_previous_examples)

    def get_pair_example_by_relative_index(self, image, relative_index):
        if relative_index < image.num_horizontal_examples:
            example = image.get_pair_example(HORIZONTAL, relative_index, self.neighbor_choices_limit)
            image_name = image.name_horizontal
        elif relative_index < image.num_examples:
            relative_index -= image.num_horizontal_examples
            example = image.get_pair_example(VERTICAL, relative_index, self.neighbor_choices_limit)
            image_name = image.name_vertical
        else:
            raise IndexError("Invalid index: {0}".format(relative_index))
        example['name'] = get_full_pair_example_name(image_name, example['part1'], example['part2'])
        return example

    def get_pair_example_by_name(self, image_name, part1, part2):
        real_pair = self.crop_pair_by_image_name(image_name, part1, part2)
        burnt_pair = self.burn_image(real_pair)
        name = get_full_pair_example_name(image_name, part1, part2)
        return {'real': real_pair, 'burnt': burnt_pair, 'name': name}



    def crop_pair_by_image_name(self, image_name, part1, part2):
        image = self._get_image_by_name(image_name)
        orientation = get_info_from_file_name(image_name, ORIENTATION_MAGIC)
        if orientation in [HORIZONTAL, VERTICAL]:
            image.crop_pair_from_image(part1, part2, orientation)
        else:
            raise KeyError("No image with valid orientation magic exists for path %s" % image_name)

    def get_image_metadata(self, image_name, image=None):
        if image is None:
            image = self._get_image_by_name(image_name)
        return image.get_metadata(image_name)

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







