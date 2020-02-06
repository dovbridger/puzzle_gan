import numpy as np
import os
from utils.plot_utils import plot_y, plot_bars
from globals import NAME_MAGIC, DELIMITER_MAGIC, PART_SIZE, ORIENTATION_MAGIC,\
    PART_SIZE_MAGIC, METADATA_FILE_NAME, HORIZONTAL, BURN_EXTENT_MAGIC


def get_info_from_file_name(file_name, requested_info_magic):
    file_name = os.path.splitext(file_name)[0]
    for info in file_name.split(DELIMITER_MAGIC):
        if info.startswith(requested_info_magic):
            return info.split(requested_info_magic)[1]
    if requested_info_magic == NAME_MAGIC:
        return os.path.splitext(file_name)[0]
    elif requested_info_magic == ORIENTATION_MAGIC:
        return HORIZONTAL


def get_full_puzzle_name_from_characteristics(puzzle_name, part_size=None, orientation=None, burn_extent=None):
    assert puzzle_name is not None and puzzle_name != '', "puzzle name must be provided"
    characteristics = [(NAME_MAGIC, puzzle_name),(PART_SIZE_MAGIC, part_size), (ORIENTATION_MAGIC, orientation),
                       (BURN_EXTENT_MAGIC, burn_extent)]
    return DELIMITER_MAGIC.join([magic + str(value) for magic, value in characteristics if value is not None])


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


def read_metadata(folder, full_puzzle_name):
    result = {}
    with open(os.path.join(folder, full_puzzle_name + DELIMITER_MAGIC + METADATA_FILE_NAME), 'r') as f:
        content = f.read().split(';')
        for item in content:
            key_value_pair = item.split(':')
            result[key_value_pair[0]] = key_value_pair[1]
    return result

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

