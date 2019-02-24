import numpy as np
import os
from utils.plot_utils import plot_y, plot_bars
from globals import NAME_MAGIC, DELIMITER_MAGIC, PART_SIZE, ORIENTATION_MAGIC,\
    PART_SIZE_MAGIC, METADATA_FILE_NAME


def get_info_from_file_name(file_name, requested_info_magic):
    for info in file_name.split(DELIMITER_MAGIC):
        if info.startswith(requested_info_magic):
            return info.split(requested_info_magic)[1]
    raise Exception("Cannot find info with magic '" + requested_info_magic + "' in file name '" + file_name + "'")


def get_full_puzzle_name_from_characteristics(puzzle_name, part_size=PART_SIZE, orientation='h'):
    return DELIMITER_MAGIC.join([NAME_MAGIC + puzzle_name,
                                 PART_SIZE_MAGIC + part_size,
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


def get_file_name_from_pair(part1, part2, orientation):
    return str(part1) + "_" + orientation + "_" + str(part2)


def get_pair_from_file_name(file_name):
    # remove extension
    file_name = file_name.split(".")[0]
    try:
        parts = file_name.split("_")
        # parts[1] is the orientation
        return int(parts[0]), int(parts[2])
    except Exception as e:
        raise Exception("Can't parse part numbers, invalid file name '" + file_name + "'. original exception: " + str(e))


def combine_all_diff_scores(model_names, correction_method='direct', indexes=range(1, 21)):
    from puzzle.scoring_statistics import COLORS, get_figure_name, run_diff_score_comparison
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


def test_compare_diff_scores(models_to_compare, indexes=range(1, 21)):
    from puzzle.scoring_statistics import run_diff_score_comparison
    for puzzle_name in [str(x) + 'b' for x in indexes]:
        print("doing puzzle " + puzzle_name)
        run_diff_score_comparison(puzzle_name,
                                  correction_method='direct',
                                  model_names=models_to_compare)


def main():
    models_to_compare = ['Original',
                         'Perfect',
                         'CalcDiffModel_g36_d30_b4',
                         'CalcDiffModel_g40_d30_b4',
                         'CalcDiffModel_g44_d20_b4',
                         'vgg_30_12']
    #test_compare_diff_scores(models_to_compare, [7])
    #combine_all_diff_scores(model_names=models_to_compare, correction_method='direct')
    from puzzle.java_utils import save_diff_matrix_cnn_for_java
    save_diff_matrix_cnn_for_java([str(x) + 'b' for x in range(1, 21)], 'CalcDiffModel_g36_d30_b4', correction_method='direct', burn_extent='4')

if __name__ == '__main__':
    main()
