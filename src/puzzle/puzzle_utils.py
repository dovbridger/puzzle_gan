import numpy as np
import os
from utils.plot_utils import plot_y, plot_bars
from globals import NAME_MAGIC, DELIMITER_MAGIC, PART_SIZE, ORIENTATION_MAGIC,\
    PART_SIZE_MAGIC, METADATA_FILE_NAME, HORIZONTAL, VERTICAL


def get_info_from_file_name(file_name, requested_info_magic):
    for info in file_name.split(DELIMITER_MAGIC):
        if info.startswith(requested_info_magic):
            return info.split(requested_info_magic)[1]
    raise Exception("Cannot find info with magic '" + requested_info_magic + "' in file name '" + file_name + "'")


def get_full_puzzle_name_from_characteristics(puzzle_name, part_size=PART_SIZE, orientation='h'):
    return DELIMITER_MAGIC.join([NAME_MAGIC + puzzle_name,
                                 PART_SIZE_MAGIC + str(part_size),
                                 ORIENTATION_MAGIC + orientation])


def matches_patterns(input_string, and_pattern, or_pattern):
    for s in and_pattern:
        if s not in input_string:
            return False
    if len(or_pattern) == 0:
        return True
    for s in or_pattern:
        if s in input_string:
            return True
    return False


def read_metadata(folder, file_name=METADATA_FILE_NAME):
    result = {}
    with open(os.path.join(folder, file_name), 'r') as f:
        content = f.read().split(';')
        for item in content:
            key_value_pair = item.split(':')
            result[key_value_pair[0]] = key_value_pair[1]
    return result


# Determine the label according the the name of the file and the number of columns in the puzzle (num_x_parts)
def determine_label(file_name, num_x_parts):
    part1, _, part2 = file_name.split('.')[0].split('_')
    part1, part2 = int(part1), int(part2)
    if part1 != part2 - 1:
        # parts are not adjacent
        return False
    # Is part1 not the last in the row
    return part1 % num_x_parts != 0


def get_full_pair_example_name(full_puzzle_name, part1, part2):
    return full_puzzle_name + "-" + str(part1) + "_" + str(part2)


def set_orientation_in_name(full_puzzle_name, orientation):
    return get_full_puzzle_name_from_characteristics(puzzle_name=get_info_from_file_name(full_puzzle_name, NAME_MAGIC),
                                                     part_size=get_info_from_file_name(full_puzzle_name, PART_SIZE_MAGIC),
                                                     orientation=orientation)

def get_pair_from_file_name(file_name):
    # remove extension
    file_name = file_name.split(".")[0]
    try:
        parts = file_name.split("_")
        # parts[1] is the orientation
        return int(parts[0]), int(parts[2])
    except Exception as e:
        raise Exception("Can't parse part numbers, invalid file name '" + file_name + "'. original exception: " + str(e))

