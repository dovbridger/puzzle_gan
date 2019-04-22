from puzzle.scoring_statistics import run_diff_score_comparison, combine_all_diff_scores,\
    make_true_false_score_histogram, plot_true_false_score_historgrams
from puzzle.puzzle_utils import get_info_from_file_name
from globals import BURN_EXTENT, NAME_MAGIC
import json
from os import listdir

LOCAL_TEST_PATH = r'C:\users\dov\checkouts\puzzle_gan_data\datasets\MET\test'


def test_compare_diff_scores(model_names, indexes=range(1, 21), correction_method='none',
                             flatten_params=''):
    for puzzle_name in [str(x) + 'b' for x in indexes]:
        print("doing puzzle " + puzzle_name)
        run_diff_score_comparison(puzzle_name,
                                  correction_method=correction_method,
                                  model_names=model_names,
                                  use_log_diff=True,
                                  flatten_params=flatten_params)

def save_diff_matrix_by_models(model_names, puzzle_names):
    from puzzle.java_utils import save_diff_matrix_cnn_for_java
    for model_name in model_names:
        save_diff_matrix_cnn_for_java(puzzle_names, model_name,
                                      additional_params='0',
                                      burn_extent=BURN_EXTENT)

def plot_probability_histogram(model_name, puzzle_names):
    true_scores, false_scores = [], []
    import numpy as np
    from puzzle.java_utils import get_probability_matrix_file_name
    for puzzle_name in puzzle_names:
        file_name = get_probability_matrix_file_name(puzzle_name, model_name)
        with open(file_name, 'r') as f:
            probability_matrix = np.array(json.load(f))
        make_true_false_score_histogram(probability_matrix, 3, 3, true_scores, false_scores)
    plot_true_false_score_historgrams(true_scores, false_scores, exclude_threshold={'True': 1,
                                                                                    'False': 0})


def main():
    test_files = listdir(LOCAL_TEST_PATH)
    puzzle_names = [get_info_from_file_name(file, NAME_MAGIC) for file in test_files if file.endswith('.png') or file.endswith('.jpg')]
    print(puzzle_names)
    models_to_compare = ['Original',
                         'Perfect',
                         'CalcProbabilityModel_g44_d25_b20_of',
                         'CalcProbabilityModel_g44_d30_b20_of']
    #plot_probability_histogram(models_to_compare[2], puzzle_names=puzzle_names)
    #combine_all_diff_scores(model_names=models_to_compare, puzzle_names=puzzle_names, use_log_diff=True, flatten_params=['', '', '', '', '', ''])
    save_diff_matrix_by_models(models_to_compare[2:3], puzzle_names=puzzle_names)


if __name__ == '__main__':
    main()