import os
import numpy as np
import json
from puzzle.puzzle_utils import get_full_puzzle_name_from_characteristics, read_metadata, get_full_pair_example_name
from globals import BURN_EXTENT, BURN_EXTENT_MAGIC, INPUT_IMAGE_TYPE, ROOT_OF_MODEL_DATA, TEST_DATA_PATH,\
    PART_SIZE

JAVA_DATA_FOLDER = r'C:\Users\dov\workspace\Puzzle Resources\Data\Init'
JAVA_DIFF_MATRIX_FILE_NAME = "diff_matrix.txt"
JAVA_MODIFIED_DIFF_MATRIX_EXTENSION = "_modified"
# One power less the the original max float (38)
MAX_FLOAT = 3.4028235e+37
INVALID_DIFF_VAL = -2
INVALID_NEIGHBOR_VAL = -1

DIFF_MATRIX_CNN_FOLDER = os.path.join(ROOT_OF_MODEL_DATA, 'diff_matrix_cnn')
PROBABILITY_MATRIX_FOLDER = os.path.join(ROOT_OF_MODEL_DATA, 'probability_matrix')

DIRECTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']

def parse_3d_numpy_array_from_json(file_name):
    with open(file_name, 'r') as f:
        result = np.array(json.load(f))
    result[result > MAX_FLOAT] = MAX_FLOAT
    return result


def save_3d_numpy_array_to_json(array, file_name):
    with open(file_name, 'w') as f:
        json.dump(array.tolist(), f)



def get_ordered_neighbors(part, diff_matrix2d):
    diff_vals = diff_matrix2d[part, :]
    sorted_indexes = list(diff_vals.argsort())
    sorted_indexes.remove(part)
    return [(index, diff_vals[index]) for index in sorted_indexes]


def get_top_k_neighbors(part, diff_matrix2d, metadata, k, reverse=False):
    if metadata['orientation'] == 'v':
        # Conversion to horizontal part numbers is needed in order to properly access the diff matrix
        part = convert_vertical_to_horizontal_part_number(part,
                                                          num_x_parts=int(metadata['num_x_parts']),
                                                          num_y_parts=int(metadata['num_y_parts']))
    k = min(k, diff_matrix2d.shape[1] - 1)
    ordered_neighbors = get_ordered_neighbors(part, diff_matrix2d)
    result = []
    for i in range(k):
        if reverse:
            i = len(ordered_neighbors) - i - 1
        result.append(ordered_neighbors[i])

    if metadata['orientation'] == 'v':
        # Conversion back to vertical part numbers is needed
        # num_x_parts and num_y_parts are reversed intentionally because of the previous conversion to horizontal
        result = [(convert_horizontal_to_vertical_part_number(part,
                                                              num_x_parts=int(metadata['num_y_parts']),
                                                              num_y_parts=int(metadata['num_x_parts'])),
                   score)
                  for (part, score) in result]
    return result


def resolve_orientation(direction, part1, part2, num_x_parts, num_y_parts):
    assert direction in [0, 1, 2, 3], "Invalid direction"
    if direction in [0, 1]:
        part1 = convert_horizontal_to_vertical_part_number(part1, num_x_parts, num_y_parts)
        part2 = convert_horizontal_to_vertical_part_number(part2, num_x_parts, num_y_parts)
    if direction in [0, 2]:
        part1, part2 = part2, part1
    return part1, part2

def convert_vertical_to_horizontal_part_number(part, num_x_parts, num_y_parts):
    '''
    Convert a part number from a vertical based image (rotated 90 degrees to the left) to a horizontal based image
    :param part: part number (counting from 0)
    :param num_x_parts: number of columns in the vertical image (rows in horizontal image)
    :param num_y_parts: number of rows in the vertical image (columns in horizontal image)
    :return: part number in the equivalent horizontal image (before the 90 degree left rotation was made)
    '''
    if part >= num_x_parts * num_y_parts or part < 0:
        raise Exception("Invalid part number " + str(part))
    row_in_vertical = int(part / num_x_parts)
    column_in_vertical = part % num_x_parts
    row_in_horizontal = column_in_vertical
    # The first row in the vertical image corresponds to the last column in the horizontal image
    column_in_horizontal = num_y_parts - 1 - row_in_vertical
    return row_in_horizontal * num_y_parts + column_in_horizontal


def convert_horizontal_to_vertical_part_number(part, num_x_parts, num_y_parts):
    '''
    Convert a part number from a horizontal based image to a vertical based image (rotated 90 degrees to the left)
    :param part: part number (counting from 0)
    :param num_x_parts: number of columns in the horizontal image (rows in vertical image)
    :param num_y_parts: number of rows in the horizontal image (columns in vertical image)
    :return: part number in the equivalent vertical image (after the 90 degree left rotation is made)
    '''
    if part >= num_x_parts * num_y_parts or part < 0:
        raise Exception("Invalid part number " + str(part))
    row_in_horizontal = int(part / num_x_parts)
    column_in_horizontal = part % num_x_parts
    # The column in the horizontal image corresponds to the last row in the vertical image
    row_in_vertical = num_x_parts - 1 - column_in_horizontal
    column_in_vertical = row_in_horizontal
    return row_in_vertical * num_y_parts + column_in_vertical


def convert_orientation_to_index(orientation):
    '''

    :param orientation: 'v' for vertical or 'h' for horizontal
    :return: corresponding index for the difference matrix
    '''
    if orientation == 'v':
        return 1
    elif orientation == 'h':
        return 3
    else:
        raise Exception("Incorrect orientation + '" + orientation + "'")


def get_java_diff_file(full_puzzle_name, burn_extent=BURN_EXTENT):
    puzzle_name, part_size, _ = full_puzzle_name.split("-")
    for file in [f for f in os.listdir(JAVA_DATA_FOLDER) if f.endswith(JAVA_DIFF_MATRIX_FILE_NAME)]:
        if puzzle_name in file and part_size in file and BURN_EXTENT_MAGIC+burn_extent in file:
            return os.path.join(JAVA_DATA_FOLDER, file)
    raise FileExistsError("Diff matrix file doesn't exist for " + full_puzzle_name)


def calc_confidence(diff_matrix3d):
    conf_matrix3d = np.zeros(diff_matrix3d.shape, dtype=diff_matrix3d.dtype)
    for orientation in range(4):
        diff_matrix2d = diff_matrix3d[orientation][:][:]
        for part in range(diff_matrix2d.shape[0]):
            # index 1 for the second item
            _, second_min_diff_value = get_ordered_neighbors(part, diff_matrix2d)[1]
            for i in range(diff_matrix2d.shape[1]):
                # default value if best neighbor and second best are both zero diff
                conf_value = float(0.001)
                if second_min_diff_value != 0:
                    conf_value = float(1) - diff_matrix2d[part][i] / second_min_diff_value
                # If the diff value is big when the conf value is positive (best neighbor)-
                # artificially lower the conf value
                if diff_matrix2d[part][i] > 1000 and conf_value > 0:
                    conf_value = conf_value / 10000
                conf_matrix3d[orientation][part][i] = conf_value
    return conf_matrix3d


def correct_invalid_values_in_matrix3d(matrix3d, correction_matrix3d, invalid_value=INVALID_DIFF_VAL,
                                       method='direct', symmetric=True):
    direction_indexes = (1, 3) if symmetric else (0, 1, 2, 3)
    for direction_index in direction_indexes:
        for part in range(matrix3d.shape[1]):
            if invalid_value in matrix3d[direction_index][part][:]:
                _correct_invalid_row(matrix3d[direction_index][part][:],
                                     correction_matrix3d[direction_index][part][:],
                                     method, invalid_value)
    if symmetric:
        _make_diff_matrix_symmetric(matrix3d)


def set_diagonal_to_value(diff_matrix, value=MAX_FLOAT):
    for i in range(4):
        for j in range(diff_matrix.shape[1]):
            diff_matrix[i][j][j] = value

def _get_mean_without_max_float(row):
    non_max_values = row[row < MAX_FLOAT]
    if non_max_values.shape[0] == 0:
        return MAX_FLOAT
    else:
        return non_max_values.mean()


def _correct_invalid_row(row, correction_row, method, invalid_value):
    invalid_value_indexes = (row == invalid_value).nonzero()[0]
    if invalid_value_indexes.shape[0] == row.shape[0]:
        row[:] = correction_row[:]
        return

    if method == 'direct':
        row[invalid_value_indexes] = correction_row[invalid_value_indexes]

    elif method == 'mean':
        valid_indexes = (row != invalid_value).nonzero()[0]
        original_mean = _get_mean_without_max_float(correction_row[valid_indexes])
        new_mean = _get_mean_without_max_float(row[valid_indexes])
        original_values = correction_row[invalid_value_indexes]
        correction_values = np.where(original_values == MAX_FLOAT,
                                     MAX_FLOAT,
                                     new_mean * (original_values / original_mean))
        row[invalid_value_indexes] = correction_values

    elif method == 'median':
        original_ranks_indexes = list(correction_row.argsort())
        new_ranks_indexes = list(row.argsort())
        # The indexes corresponding to invalid values will be at the beginning and we want to remove them
        new_ranks_indexes = new_ranks_indexes[invalid_value_indexes.shape[0]:]

        # 'positions' in original_ranks_indexes
        original_invalid_ranks_indexes_and_positions = [(original_ranks_indexes.index(x), x) for x in invalid_value_indexes]
        original_invalid_ranks_indexes_and_positions.sort()
        for position, rank_index in original_invalid_ranks_indexes_and_positions:
            new_ranks_indexes.insert(position, rank_index)

        current_rank = 0
        lower_valid_value = None
        while current_rank < len(new_ranks_indexes):
            if new_ranks_indexes[current_rank] not in invalid_value_indexes:
                lower_valid_value = row[new_ranks_indexes[current_rank]]
                current_rank = current_rank + 1
                continue
            else:
                num_consecutive_invalid_values = 0
                while current_rank < len(new_ranks_indexes) and new_ranks_indexes[current_rank] in invalid_value_indexes:
                    num_consecutive_invalid_values = num_consecutive_invalid_values + 1
                    current_rank = current_rank + 1
                upper_valid_value = None if current_rank == len(new_ranks_indexes) else\
                    row[new_ranks_indexes[current_rank]]
                if num_consecutive_invalid_values > 0:
                    correction_values = assign_diff_values_by_original_rank(num_consecutive_invalid_values, lower_valid_value, upper_valid_value)
                    for i in range(1, num_consecutive_invalid_values + 1):
                        row[new_ranks_indexes[current_rank - i]] = correction_values[-i]

    else:
        raise Exception("unknown correction method '" + method + "'")


def _make_diff_matrix_symmetric(diff_matrix3d):
    '''
    Make diff_matrix symmetric - make orientations up(0) and left(2) equal to down(1) and right(3)
    '''
    num_parts = diff_matrix3d.shape[1]
    for direction, opposite in [(1, 0), (3, 2)]:
        for part1 in range(num_parts):
            for part2 in range(num_parts):
                # Make symmetric diff
                diff_matrix3d[opposite][part1][part2] = diff_matrix3d[direction][part2][part1]


def assign_diff_values_by_original_rank(num_values, lower_bound=None, upper_bound=None):
    result = np.zeros(num_values, dtype='float32')
    if lower_bound is None:
        result[:] = upper_bound# - 0.01 * np.arange(num_values, 0, -1)
    elif upper_bound is None:
        result[:] = lower_bound# + 0.01 * np.arange(num_values, 0, -1)
    else:
        coefficients = (np.arange(num_values, dtype='float32') + 1) / (num_values + 1)
        result = lower_bound + coefficients * (upper_bound - lower_bound)
    result[result < 0] = 0
    return result


def save_diff_matrix_cnn_for_java(puzzle_names, model_name, correction_method='direct', burn_extent=BURN_EXTENT):
    for puzzle_name in puzzle_names:
        diff_matrix_cnn = load_diff_matrix_cnn(puzzle_name, model_name)
        full_puzzle_name = get_full_puzzle_name_from_characteristics(puzzle_name)
        original_java_diff_matrix_file = get_java_diff_file(full_puzzle_name, burn_extent=burn_extent)
        diff_matrix_original = parse_3d_numpy_array_from_json(original_java_diff_matrix_file)
        correct_invalid_values_in_matrix3d(diff_matrix_cnn, diff_matrix_original, method=correction_method)
        modified_java_diff_matrix_file = original_java_diff_matrix_file[:-4] +\
                                         JAVA_MODIFIED_DIFF_MATRIX_EXTENSION + "_" + correction_method +\
                                         original_java_diff_matrix_file[-4:]
        save_3d_numpy_array_to_json(diff_matrix_cnn, modified_java_diff_matrix_file)


def parse_java_scores(file_name):
    with open(file_name, 'r') as f:
        content = f.read()
    content = content.replace('\n', '')
    lines = content.split('name: ')[1:]
    scores = {}
    for line in lines:
        name = line.split(" ")[0]
        if name not in scores:
            scores[name] = []
        score = int(line.split("score = ")[1])
        method = line.split("methods=")[1]
        method = method[:method.index(' ')]
        scores[name].append((score, method))
    return scores


def create_diff_matrix2d_with_model_evaluations(model, dataset, metadata):

    num_x_parts = metadata.num_x_parts
    num_y_parts = metadata.num_y_parts
    num_parts = num_x_parts * num_y_parts
    diff_matrix2d = np.zeros((num_parts, num_parts), dtype='float32')
    diff_matrix2d[:][:] = INVALID_DIFF_VAL
    for part1 in range(num_parts):
        print("Orientation {0}, part1 {1}".format(metadata.orientation, part1, num_x_parts))
        for part2 in range(num_parts):
            if part1 == part2:
                continue
            example_data = dataset.get_pair_example_by_name(metadata.full_puzzle_name, part1, part2)
            prediction = model.predict(example_data)

            if metadata.orientation == 'v':
                # Conversion to horizontal needed
                part1_horizontal = convert_vertical_to_horizontal_part_number(part1, num_x_parts=num_x_parts, num_y_parts=num_y_parts)
                part2_horizontal = convert_vertical_to_horizontal_part_number(part2, num_x_parts=num_x_parts, num_y_parts=num_y_parts)
            else:
                part1_horizontal, part2_horizontal = part1, part2

            min_prediction = float(1) / MAX_FLOAT
            if prediction < min_prediction:
                diff_matrix2d[part1_horizontal][part2_horizontal] = MAX_FLOAT
            else:
                diff_matrix2d[part1_horizontal][part2_horizontal] = (float(1) / prediction) - 1
    return diff_matrix2d


def create_probability_matrix2d_with_model_evaluations(model, dataset, metadata):
    num_x_parts = metadata.num_x_parts
    num_y_parts = metadata.num_y_parts
    num_parts = num_x_parts * num_y_parts
    probability_matrix2d = np.zeros((num_parts, num_parts), dtype='float32')
    probability_matrix2d[:][:] = INVALID_DIFF_VAL
    for part1 in range(num_parts):
        print("Orientation {0}, part1 {1}".format(metadata.orientation, part1, num_x_parts))
        for part2 in range(num_parts):
            if part1 == part2:
                continue
            example_data = dataset.get_pair_example_by_name(metadata.full_puzzle_name, part1, part2)
            prediction = model.predict(example_data)

            if metadata.orientation == 'v':
                # Conversion to horizontal needed
                part1_horizontal = convert_vertical_to_horizontal_part_number(part1, num_x_parts=num_x_parts, num_y_parts=num_y_parts)
                part2_horizontal = convert_vertical_to_horizontal_part_number(part2, num_x_parts=num_x_parts, num_y_parts=num_y_parts)
            else:
                part1_horizontal, part2_horizontal = part1, part2

            probability_matrix2d[part1_horizontal][part2_horizontal] = prediction
    return probability_matrix2d


def get_diff_matrix_cnn_file_name(puzzle_name, model_name):
    return os.path.join(DIFF_MATRIX_CNN_FOLDER, model_name + "-" + puzzle_name + ".json")


def get_probability_matrix_file_name(puzzle_name, model_name):
    return os.path.join(PROBABILITY_MATRIX_FOLDER, model_name + "-" + puzzle_name + ".json")


def load_diff_matrix_cnn(puzzle_name, model_name, model=None, file_name=None):
    try:
        if file_name is None:
            file_name = get_diff_matrix_cnn_file_name(puzzle_name, model_name)
        with open(file_name, 'r') as f:
            diff_matrix_cnn = np.array(json.load(f))
    except Exception as e:
        print("Could not load diff_matrix_cnn from json file '" + file_name + "'")
        print("Exception: " + str(e))
        if model is None:
            print("No model was provided, so diff_matrix_cnn cannot be created either")
            diff_matrix_cnn = None
        else:
            print("Creating diff_matrix_cnn with model evaluations")
            diff_matrix_cnn = create_diff_matrix3d_with_model_evaluations(puzzle_name, model_name, model)

    return diff_matrix_cnn


def load_diff_matrix_cnn_from_probability(puzzle_name, model_name, file_name=None, use_log=False):
    if file_name is None:
        file_name = get_probability_matrix_file_name(puzzle_name, model_name)
    with open(file_name, 'r') as f:
        probability_matrix = np.array(json.load(f))
    return convert_probability_matrix_to_diff(probability_matrix, use_log)


def create_diff_matrix3d_with_model_evaluations(puzzle_name, part_size, model, dataset):
    diff_matrix3d = None
    for orientation, direction_index in [('h', 3), ('v', 1)]:
        full_puzzle_name = get_full_puzzle_name_from_characteristics(puzzle_name=puzzle_name,
                                                                     part_size=str(part_size),
                                                                     orientation=orientation)
        metadata = dataset.get_image_metadata(full_puzzle_name)
        diff_matrix2d = create_diff_matrix2d_with_model_evaluations(model, dataset, metadata)
        # For the first iteration
        if diff_matrix3d is None:
            num_parts = diff_matrix2d.shape[0]
            diff_matrix3d = np.zeros((4, num_parts, num_parts), dtype=diff_matrix2d.dtype)

        diff_matrix3d[direction_index][:][:] = diff_matrix2d

    _make_diff_matrix_symmetric(diff_matrix3d)
    set_diagonal_to_value(diff_matrix3d)
    try:
        json_file_name = get_diff_matrix_cnn_file_name(puzzle_name, model.instance_name())
        with open(json_file_name, 'w') as f:
            json.dump(diff_matrix3d.tolist(), f)
    except Exception as e:
        print("Could not save diff_matrix_cnn to json file '" + json_file_name + "'")
        print("Exception: " + str(e))
        pass
    return diff_matrix3d


def create_probability_matrix3d_with_model_evaluations(puzzle_name, part_size, model, dataset):
    probability_matrix3d = None
    for orientation, direction_index in [('h', 3), ('v', 1)]:
        full_puzzle_name = get_full_puzzle_name_from_characteristics(puzzle_name=puzzle_name,
                                                                     part_size=str(part_size),
                                                                     orientation=orientation)
        metadata = dataset.get_image_metadata(full_puzzle_name)
        probability_matrix2d = create_probability_matrix2d_with_model_evaluations(model, dataset, metadata)
        # For the first iteration
        if probability_matrix3d is None:
            num_parts = probability_matrix2d.shape[0]
            probability_matrix3d = np.zeros((4, num_parts, num_parts), dtype=probability_matrix2d.dtype)

        probability_matrix3d[direction_index][:][:] = probability_matrix2d

    _make_diff_matrix_symmetric(probability_matrix3d)
    set_diagonal_to_value(probability_matrix3d, value=0)
    try:
        json_file_name = get_probability_matrix_file_name(puzzle_name, model.instance_name())
        with open(json_file_name, 'w') as f:
            json.dump(probability_matrix3d.tolist(), f)
    except Exception as e:
        print("Could not save probability_matrix to json file '" + json_file_name + "'")
        print("Exception: " + str(e))
        pass
    return probability_matrix3d


def convert_probability_matrix_to_diff(probability_matrix3d, use_log):

    if use_log:
        diff_matrix3d = np.where(probability_matrix3d == 0, MAX_FLOAT, -np.log(probability_matrix3d))
    else:
        min_prediction = float(1) / MAX_FLOAT
        diff_matrix3d = np.where(probability_matrix3d <= min_prediction,
                                 MAX_FLOAT,
                                 (float(1) / probability_matrix3d) - 1)
    return diff_matrix3d

