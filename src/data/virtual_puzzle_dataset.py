import os.path
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
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
        self.transform = get_transform(opt)
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

    def get_pair_example_by_index(self, index):
        relevant_image_index = bisect([x.num_examples_accumulated for x in self.images], index)
        relevant_image = self.images[relevant_image_index]
        num_previous_examples = relevant_image.num_examples_accumulated - relevant_image.num_examples
        return self.get_pair_example_by_relative_index(relevant_image, index - num_previous_examples)

    def get_pair_example_by_relative_index(self, image, relative_index):
        if relative_index < image.num_horizontal_examples:
            example = self.get_pair_example_from_specific_image(image.horizontal, image.num_x_parts,
                                                                image.neighbor_choices[HORIZONTAL], relative_index)
            image_name = image.name_horizontal
        elif relative_index < image.num_examples:
            relative_index -= image.num_horizontal_examples
            # num_y_parts is the number of x parts in the vertical image
            example = self.get_pair_example_from_specific_image(image.vertical, image.num_y_parts,
                                                                image.neighbor_choices[VERTICAL], relative_index)
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

    def get_pair_example_from_specific_image(self, specific_image, num_x_parts, neighbor_choices, relative_index):
        # The part that corresponds with relative_index
        part_index = int(relative_index / (self.opt.num_false_examples + 1))
        # The part in the original puzzle (after skipping the rightmost column)
        part1 = int(part_index * num_x_parts / (num_x_parts - 1))
        assert part1 % num_x_parts < num_x_parts - 1, "part1 was selected from rightmost column, sum tin wong"
        # Initialize part2 as the true neighbor
        part2 = part1 + 1
        if relative_index % (self.opt.num_false_examples + 1) == 0:
            # A true example
            label = 1
        else:
            label = 0
            while (part2 == part1 + 1 or part2 == part1):
                # Randomly select a part until it is valid
                part2 = choice(neighbor_choices[part1][:self.neighbor_choices_limit])
 #               print("chose {0} for {1] from {1}".format(part2, part1, neighbor_choices[part1][:self.neighbor_choices_limit]))
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

class VirtualImage():
    def __init__(self, path, opt, num_examples_accumulated):
        self.image_dir = os.path.dirname(path)
        self.name_horizontal, self.image_extension = os.path.splitext(os.path.basename(path))
        self.name_vertical = set_orientation_in_name(self.name_horizontal, VERTICAL)

        self.horizontal = VirtualPuzzleDataset.get_real_image(os.path.join(
            self.image_dir,
            self.name_horizontal + self.image_extension))
        self.vertical = self.horizontal.transpose(2, 1).flip(1)
        _, height, width = self.horizontal.shape

        assert height % opt.part_size == 0 and width % opt.part_size == 0, \
            "Image ({0}x{1} wasn't cropped to be a multiple of 'part_size'({2})".format(height, width,
                                                                                        opt.part_size)
        self.num_x_parts = int(width / opt.part_size)
        self.num_y_parts = int(height / opt.part_size)
        self.num_horizontal_examples = self.count_pair_examples_in_image(self.num_x_parts,
                                                                         self.num_y_parts,
                                                                         opt.num_false_examples)
        # x and y are reversed in vertical
        self.num_vertical_examples = self.count_pair_examples_in_image(self.num_y_parts,
                                                                       self.num_x_parts,
                                                                       opt.num_false_examples)
        self.num_examples = self.num_horizontal_examples + self.num_vertical_examples

        self.num_examples_accumulated = self.num_examples + num_examples_accumulated
        self.neighbor_choices = self.get_neighbor_choices(opt)

    def get_neighbor_choices(self, opt):
        num_parts = self.num_x_parts * self.num_y_parts
        if opt.max_neighbor_rank <= 1:
            all_choices = [range(num_parts) for part in range(num_parts)]
            return {HORIZONTAL: all_choices, VERTICAL: all_choices}

        puzzle_name = get_info_from_file_name(self.name_horizontal, NAME_MAGIC)
        original_diff_matrix3d = parse_3d_numpy_array_from_json(get_java_diff_file(puzzle_name,
                                                                                   burn_extent='0',
                                                                                   part_size=opt.part_size))
        result = {HORIZONTAL: [], VERTICAL: []}
        for name in [self.name_horizontal, self.name_vertical]:
            metadata = self.get_metadata(name)
            direction_index = convert_orientation_to_index(metadata.orientation)
            diff_matrix2d = original_diff_matrix3d[direction_index, :, :]
            for part in range(num_parts):
                current_part_choices = [part for (part, score) in
                                        get_top_k_neighbors(part, diff_matrix2d, metadata,
                                                            opt.max_neighbor_rank, reverse=False)]
                adjacent = part + 1

                # Remove true neighbor from that part's false neighbor choices (if exists)
                if adjacent % metadata.num_x_parts != 0 and adjacent in current_part_choices:
                    current_part_choices.remove(adjacent)

                # Remove the part itself from it's neighbor choices
                if part in current_part_choices:
                    current_part_choices.remove(part)
                result[metadata.orientation].append(current_part_choices)
        return result

    def get_metadata(self, image_name):
        metadata = Namespace()
        metadata.orientation = get_info_from_file_name(image_name, ORIENTATION_MAGIC)
        assert metadata.orientation in [HORIZONTAL, VERTICAL], "Invalid orientation in image_name"
        metadata.full_puzzle_name = image_name
        if metadata.orientation == HORIZONTAL:
            metadata.num_x_parts = self.num_x_parts
            metadata.num_y_parts = self.num_y_parts
        elif metadata.orientation == VERTICAL:
            metadata.num_x_parts = self.num_y_parts
            metadata.num_y_parts = self.num_x_parts
        return metadata

    @staticmethod
    def count_pair_examples_in_image(num_x_parts, num_y_parts, num_false_examples):
        num_true_examples = num_y_parts * (num_x_parts - 1)
        return num_true_examples * (num_false_examples + 1)







