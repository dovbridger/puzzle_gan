import os
import torch
from random import choice
from argparse import Namespace
from puzzle.puzzle_utils import set_orientation_in_name, get_info_from_file_name
from puzzle.java_utils import get_top_k_neighbors, get_java_diff_file, parse_3d_numpy_array_from_json,\
    convert_orientation_to_index
from globals import HORIZONTAL, VERTICAL, NAME_MAGIC, ORIENTATION_MAGIC
from data.virtual_puzzle_dataset import VirtualPuzzleDataset
class VirtualImage:
    def __init__(self, path, opt, num_examples_accumulated):
        self.image_dir = os.path.dirname(path)
        self.name_horizontal, self.image_extension = os.path.splitext(os.path.basename(path))
        self.name_vertical = set_orientation_in_name(self.name_horizontal, VERTICAL)
        self.opt = opt

        self.horizontal = VirtualPuzzleDataset.get_real_image(os.path.join(
            self.image_dir,
            self.name_horizontal + self.image_extension))
        self.vertical = self.horizontal.transpose(2, 1).flip(1)
        self.horizontal_parts = self.get_parts_array(self.horizontal)
        _, height, width = self.horizontal.shape
        assert height % opt.part_size == 0 and width % self.part_size == 0, \
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

    def get_neighbor_choices(self):
        num_parts = self.num_x_parts * self.num_y_parts
        if self.opt.max_neighbor_rank <= 1:
            all_choices = [range(num_parts) for part in range(num_parts)]
            return {HORIZONTAL: all_choices, VERTICAL: all_choices}

        puzzle_name = get_info_from_file_name(self.name_horizontal, NAME_MAGIC)
        original_diff_matrix3d = parse_3d_numpy_array_from_json(get_java_diff_file(puzzle_name,
                                                                                   burn_extent='0',
                                                                                   part_size=self.opt.part_size))
        result = {HORIZONTAL: [], VERTICAL: []}
        for name in [self.name_horizontal, self.name_vertical]:
            metadata = self.get_metadata(name)
            direction_index = convert_orientation_to_index(metadata.orientation)
            diff_matrix2d = original_diff_matrix3d[direction_index, :, :]
            for part in range(num_parts):
                current_part_choices = [part for (part, score) in
                                        get_top_k_neighbors(part, diff_matrix2d, metadata,
                                                            self.opt.max_neighbor_rank, reverse=False)]
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

    def crop_part_from_image(self, image_array, num_columns, part):
        row = int(part / num_columns)
        column = part % num_columns
        return image_array[:, row * self.opt.part_size: (row + 1) * self.opt.part_size,
                           column * self.opt.part_size: (column + 1) * self.opt.part_size]

    def crop_pair_from_image(self, part1, part2, orientation):

        part1_tensor = self.get_part_from_image(part1, orientation)
        part2_tensor = self.get_part_from_image(part2, orientation)
        return torch.cat((part1_tensor, part2_tensor), 2)

    def create_individaul_parts(self):
        self.individual_parts = {HORIZONTAL: self.get_parts_array(self.horizontal, self.num_x_parts),
                                 VERTICAL: self.get_parts_array(self.vertical, self.num_y_parts)}
        self.horizontal, self.vertical = None, None

    def get_parts_array(self, image_array, num_columns):
        num_parts = self.num_x_parts * self.num_y_parts
        all_parts = tuple(self.crop_part_from_image(image_array, part, num_columns).unsqueeze(0)
                          for part in range(num_parts))
        return torch.cat(all_parts, dim=0)

    def get_part_from_image(self, part, orientation):
        return self.individual_parts[orientation][part, :, :, :]

    def get_pair_example(self, orientation, relative_index):
        num_columns = self.num_x_parts if orientation is HORIZONTAL else self.num_y_parts
        neighbor_choices = self.neighbor_choices[orientation]
        # The part that corresponds with relative_index
        part_index = int(relative_index / (self.opt.num_false_examples + 1))
        # The part in the original puzzle (after skipping the rightmost column)
        part1 = int(part_index * num_columns / (num_columns - 1))
        assert part1 % num_columns < num_columns - 1, "part1 was selected from rightmost column, sum tin wong"
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
        pair_tensor = self.crop_pair_from_image(part1, part2, orientation)
        return {'part1': part1, 'part2': part2, 'label': label, 'real': pair_tensor}