import numpy as np
from os import path
from puzzle.java_utils import get_ordered_neighbors, correct_invalid_values_in_matrix3d, INVALID_NEIGHBOR_VAL,\
    get_java_diff_file, parse_3d_numpy_array_from_json, calc_confidence, load_diff_matrix_cnn, resolve_orientation
from puzzle.puzzle_utils import get_full_puzzle_name_from_characteristics, read_metadata
from utils.plot_utils import plot_y, plot_bars, plot_images
from globals import TEST_DATA_PATH, FIGURES_FOLDER, PART_SIZE
from data.virtual_puzzle_dataset import VirtualPuzzleDataset
from argparse import Namespace

ORIGINAL_DIFF_MATRIX_NAME = 'Original'
PERFECT_DIFF_MATRIX_NAME = 'Perfect'
DATA_LABELS = [ORIGINAL_DIFF_MATRIX_NAME, PERFECT_DIFF_MATRIX_NAME]
COLORS = ['b', 'r', 'y', 'g', 'm', 'k', 'c']


def get_true_neighbor(part, direction, num_x_parts, num_y_parts):
    '''
    :param part: the part number
    :param direction: 0-up, 1-down, 2-left ,3-right
    :return: The correct neighbor for a piece "part" direction "direction" given the puzzle dimensions
    '''

    if direction == 0:
        if part >= num_x_parts:
            return part - num_x_parts

    if direction == 1:
        if part < num_x_parts * (num_y_parts - 1):
            return part + num_x_parts

    if direction == 2:
        if part % num_x_parts > 0:
            return part - 1

    if direction == 3:
        if part % num_x_parts < num_x_parts - 1:
            return part + 1

    return INVALID_NEIGHBOR_VAL


def get_rank_statistics(diff_matrix3d, num_x_parts, num_y_parts):
    ranks = np.zeros((4, num_x_parts * num_y_parts), dtype='int32')
    ranks[:][:] = INVALID_NEIGHBOR_VAL
    for direction in range(4):
        for part in range(num_x_parts * num_y_parts):
            true_neighbor = get_true_neighbor(part, direction, num_x_parts, num_y_parts)
            if true_neighbor != INVALID_NEIGHBOR_VAL:
                ordered_neighbors = [index for (index, score) in
                                     get_ordered_neighbors(part, diff_matrix3d[direction][:][:])]
                for i in range(len(ordered_neighbors)):
                    if ordered_neighbors[i] == true_neighbor:
                        ranks[direction][part] = i
                        break
                if ranks[direction][part] == INVALID_NEIGHBOR_VAL:
                    raise Exception("No rank was found for part " + str(part) + " in direction " + str(direction))
    return ranks


def flatten_and_remove_invalid_ranks(ranks):
    return np.array([val for val in ranks.flatten() if val != INVALID_NEIGHBOR_VAL])

def split_conf_scores_to_true_and_false(conf_matrix, num_x_parts, num_y_parts, conf_score_identities=None):
    num_parts = conf_matrix.shape[1]
    true_neighbor_top_scores = []
    false_neighbor_top_scores = []
    for direction in range(4):
        true_neighbor_top_scores.append([])
        false_neighbor_top_scores.append([])
        top_scores = np.array([get_ordered_neighbors(part, conf_matrix[direction][:][:])[-1]
                               for part in range(num_parts)])
        for part in range(num_parts):
            neighbor, score = top_scores[part]
            if neighbor == get_true_neighbor(part, direction, num_x_parts, num_y_parts):
                true_neighbor_top_scores[direction].append(score)
            else:
                false_neighbor_top_scores[direction].append(score)
                if conf_score_identities is not None:
                    conf_score_identities.append((score, (direction, part, neighbor)))
    return [true_neighbor_top_scores, false_neighbor_top_scores]


def count_best_buddies(diff_matrix3d, num_x_parts, num_y_parts):
    '''

    :param diff_matrix3d:
    :param num_x_parts:
    :param num_y_parts:
    :return: numpy.array([[True best buddies Horizontal , True Best Buddies Vertical]
                          [False best buddies Horizontal , False Best Buddies Vertical]]
    Where 'True best buddies' and 'False Best Buddies' are numpy arrays containing 2 values - horizontal best buddies
    count and vertical
    '''
    num_parts = diff_matrix3d.shape[1]
    true_best_buddies = np.zeros((2,), dtype='int32')
    false_best_buddies = np.zeros((2,), dtype='int32')
    directions = [(1, 0), (3, 2)]
    for i in range(len(directions)):
        direction, reversed_direction = directions[i]
        for part in range(num_parts):
            best_neighbor = diff_matrix3d[direction][part][:].argsort()[0]
            if part == diff_matrix3d[reversed_direction][best_neighbor][:].argsort()[0]:
                if best_neighbor == get_true_neighbor(part, direction, num_x_parts, num_y_parts):
                    true_best_buddies[i] = true_best_buddies[i] + 1
                else:
                    false_best_buddies[i] = false_best_buddies[i] + 1
    return np.array((true_best_buddies, false_best_buddies))


def _load_diff_matricies_for_comparison(puzzle_name, model_names):
    full_puzzle_name = get_full_puzzle_name_from_characteristics(puzzle_name, orientation='h')
    metadata = read_metadata(path.join(TEST_DATA_PATH, full_puzzle_name))
    diff_matricies = []
    for model_name in model_names:
        if model_name == ORIGINAL_DIFF_MATRIX_NAME:
            diff_matrix = parse_3d_numpy_array_from_json(get_java_diff_file(full_puzzle_name))
        elif model_name == PERFECT_DIFF_MATRIX_NAME:
            java_diff_file_perfect = get_java_diff_file(full_puzzle_name, burn_extent='0')
            diff_matrix = parse_3d_numpy_array_from_json(java_diff_file_perfect)
        else:
            diff_matrix = load_diff_matrix_cnn(puzzle_name, model_name)
        diff_matricies.append(diff_matrix)
    return diff_matricies, metadata


def _flatten_uneven_nested_list(array):
    result = []
    for i in range(len(array)):
        result = result + array[i]
    return np.array(result)


def _get_conf_scores(diff_matrix, num_x_parts, num_y_parts):
    conf_matrix = calc_confidence(diff_matrix)
    # [true_scores, false_scores]
    scores = split_conf_scores_to_true_and_false(conf_matrix,
                                                 num_x_parts=num_x_parts,
                                                 num_y_parts=num_y_parts)
    return scores


def run_diff_score_comparison(puzzle_name, correction_method='none', model_names=DATA_LABELS):
    diff_matrices, metadata = _load_diff_matricies_for_comparison(puzzle_name, model_names)
    # Correct the cnn diff matrix with the original diff matrix
    for i in range(len(model_names)):
        if model_names[i] not in [ORIGINAL_DIFF_MATRIX_NAME, PERFECT_DIFF_MATRIX_NAME]:
            if correction_method != 'none':
                correct_invalid_values_in_matrix3d(diff_matrices[i],
                                                   diff_matrices[model_names.index(ORIGINAL_DIFF_MATRIX_NAME)],
                                                   method=correction_method)
    return compare_diff_scores(puzzle_name, diff_matrices,
                               metadata, correction_method,
                               diff_matrix_names=model_names)


def compare_diff_scores(puzzle_name, diff_matricies, metadata, correction_method, diff_matrix_names):
    all_data_out = {}
    num_x_parts = int(metadata['num_x_parts'])
    num_y_parts = int(metadata['num_y_parts'])
    description = puzzle_name + '-' + correction_method
    all_data_out['ranks'] = compare_ranks(diff_matricies,
                                          num_x_parts=num_x_parts,
                                          num_y_parts=num_y_parts,
                                          puzzle_description=description,
                                          diff_matrix_names=diff_matrix_names)
    all_data_out['buddies'] = compare_best_buddies(diff_matricies,
                                                   num_x_parts=num_x_parts,
                                                   num_y_parts=num_y_parts,
                                                   puzzle_description=description,
                                                   diff_matrix_names=diff_matrix_names)
    all_data_out['confidence'] = compare_confidence_scores(diff_matricies,
                                                           num_x_parts=num_x_parts,
                                                           num_y_parts=num_y_parts,
                                                           puzzle_description=description,
                                                           diff_matrix_names=diff_matrix_names)
    return all_data_out


def compare_ranks(diff_matricies, num_x_parts, num_y_parts, puzzle_description, diff_matrix_names):
    data_to_plot = []
    for diff_matrix in diff_matricies:
        ranks = flatten_and_remove_invalid_ranks(get_rank_statistics(diff_matrix,
                                                                     num_x_parts=num_x_parts,
                                                                     num_y_parts=num_y_parts))
        data_to_plot.append(ranks)
    data_to_plot = tuple(data_to_plot)
    plot_y(data_to_plot,
           sort=True, colors=COLORS, labels=diff_matrix_names,
           titles=['Diff score ranking of the true neighbors (sorted)'],
           output_file_name=get_figure_name(puzzle_description, 'ranks'))
    return data_to_plot


def compare_confidence_scores(diff_matrices, num_x_parts, num_y_parts, puzzle_description, diff_matrix_names):
    plot_titles = ['Top True Scores', 'Top False scores']
    # Zip original true with cnn true, and original false with cnn false
    data_to_plot = [
        # Flatten scores across the 4 directions
        [_flatten_uneven_nested_list(scores) for scores in _get_conf_scores(diff_matrix, num_x_parts, num_y_parts)]
        for diff_matrix in diff_matrices
    ]

    # Zip diff_matrix1 true with diff_matrix2 true, and diff_matrix1 false with diff_matrix2 false
    data_to_plot = list(zip(*data_to_plot))
    plot_y(data_to_plot,
           sort=True, colors=COLORS, labels=diff_matrix_names,
           titles=plot_titles, output_file_name=get_figure_name(puzzle_description, 'confidence_scores'))
    return data_to_plot


def compare_best_buddies(diff_matrices, num_x_parts, num_y_parts, puzzle_description, diff_matrix_names=DATA_LABELS):
    best_buddies_counts = [count_best_buddies(diff_matrix3d, num_x_parts=num_x_parts, num_y_parts=num_y_parts) for
                          diff_matrix3d in diff_matrices]

    plot_titles = ['Total Best Buddies Count']
    tick_labels = ['True BB', 'False BB']

    data_to_plot = tuple(best_buddies_count.sum(axis=1)for best_buddies_count in best_buddies_counts)
    plot_bars(data_to_plot, labels=diff_matrix_names, colors=COLORS,
              titles=plot_titles, tick_labels=tick_labels, output_file_name=get_figure_name(puzzle_description, 'best_buddies'))
    return data_to_plot


def combine_all_diff_scores(model_names, correction_method='direct', indexes=range(1, 21)):
    puzzle_names = [str(x) + 'b' for x in indexes]
    all_scores = {'ranks': [], 'buddies': [], 'confidence': []}
    for puzzle_name in puzzle_names:
        print("Collecting scores for puzzle " + puzzle_name)
        current_scores = run_diff_score_comparison(puzzle_name, correction_method, model_names=model_names)
        for key in current_scores:
            all_scores[key].append(current_scores[key])
    print("Collected all scores")
    buddies = tuple(x for x in np.array(all_scores['buddies']).sum(axis=0))
    tick_labels = ['True BB', 'False BB']
    ranks = np.array(all_scores['ranks']).transpose(([1, 2, 0]))
    ranks = tuple(ranks.reshape(ranks.shape[0], -1))
    confidence = list(zip(*all_scores['confidence']))
    for i in range(2):
        score_sums = [() for k in range(len(confidence[i][0]))]
        for puzzle_scores in confidence[i]:
            puzzle_scores = list(puzzle_scores)
            for k in range(len(puzzle_scores)):
                score = tuple(puzzle_scores[k])
                score_sums[k] = score_sums[k] + score
        confidence[i] = tuple(np.array(score_sum).flatten() for score_sum in score_sums)
    puzzle_description = "All_puzzles-" + correction_method
    plot_y(ranks,
           sort=True, colors=COLORS, labels=model_names,
           titles=['Diff score ranking of the true neighbors (sorted)'],
           output_file_name=get_figure_name(puzzle_description, 'ranks'))
    plot_bars(buddies, labels=model_names, colors=COLORS,
              titles=['Total Best Buddies Count'], tick_labels=tick_labels,
              output_file_name=get_figure_name(puzzle_description, 'best_buddies'))
    plot_y(confidence,
           sort=True, colors=COLORS, labels=model_names,
           titles=['Top True Scores', 'Top False scores'], output_file_name=get_figure_name(puzzle_description, 'confidence_scores'))


def get_top_false_confidence_score_identities(conf_matrix, num_x_parts, num_y_parts, num_results):
    false_score_identities = []
    _,_ = split_conf_scores_to_true_and_false(conf_matrix, num_x_parts, num_y_parts, false_score_identities)
    false_score_identities = sorted(false_score_identities)
    false_score_identities.reverse()
    return false_score_identities[0: num_results]


def visualize_confidence_mistakes(puzzle_name, model_name, num_results=1):
    direction_names = ['Up', 'Down', 'Left', 'Right']
    x_parts, y_parts = 15, 10
    diff_matrix = load_diff_matrix_cnn(puzzle_name, model_name)
    top_mistakes_identities = get_top_false_confidence_score_identities(calc_confidence(diff_matrix), x_parts, y_parts, num_results)
    opt = Namespace
    opt.dataroot = r'C:\SHARE\checkouts\puzzle_gan_data\datasets\virtual_puzzle_parts'
    opt.phase = 'test'
    opt.part_size = 64
    opt.puzzle_name = puzzle_name
    opt.num_false_examples = 1
    opt.burn_extent = 4
    dataset = VirtualPuzzleDataset()
    dataset.initialize(opt)
    images = []
    titles = []
    image_name_horizontal = get_full_puzzle_name_from_characteristics(puzzle_name, opt.part_size)
    image_name_vertical = image_name_horizontal.replace('orientation=h', 'orientation=v')
    for score, (direction, part1, part2) in top_mistakes_identities:
        part2 = int(part2)
        direction, part1, part2 = resolve_orientation(direction, part1, part2, num_x_parts=15, num_y_parts=10)
        image_name = image_name_horizontal if direction in [2, 3] else image_name_vertical
        pair_example = dataset.get_pair_numpy(image_name, part1, part2)
        images.append(pair_example)
        titles.append("{0}-{1}_{2} : {3}".format(direction_names[direction], part1, part2, round(score, 2)))
    plot_images(np.array(images), titles=titles, figsize=(36, 18),
                output_file_name=get_figure_name(puzzle_name, 'top_mistakes'))


def get_figure_name(puzzle_description, figure_type):
    return path.join(FIGURES_FOLDER, puzzle_description + "-" + figure_type + '.jpg')






