from puzzle.scoring_statistics import run_diff_score_comparison, combine_all_diff_scores,\
    make_true_false_score_histogram, plot_true_false_score_historgrams
from globals import BURN_EXTENT
import json


def test_compare_diff_scores(model_names, indexes=range(1, 21), correction_method='none',
                             flatten_params=''):
    for puzzle_name in [str(x) + 'b' for x in indexes]:
        print("doing puzzle " + puzzle_name)
        run_diff_score_comparison(puzzle_name,
                                  correction_method=correction_method,
                                  model_names=model_names,
                                  use_log_diff=True,
                                  flatten_params=flatten_params)

def save_diff_matrix_by_models(model_names):
    from puzzle.java_utils import save_diff_matrix_cnn_for_java
    for model_name in model_names:
        save_diff_matrix_cnn_for_java([str(x) + 'b' for x in range(1, 21)], model_name,
                                      additional_params='0',
                                      burn_extent=BURN_EXTENT)

def plot_probability_histogram(model_name, indexes=range(1, 21)):
    true_scores, false_scores = [], []
    import numpy as np
    from puzzle.java_utils import get_probability_matrix_file_name
    for puzzle_name in [str(x) + 'b' for x in indexes]:
        file_name = get_probability_matrix_file_name(puzzle_name, model_name)
        with open(file_name, 'r') as f:
            probability_matrix = np.array(json.load(f))
        make_true_false_score_histogram(probability_matrix, 15, 10, true_scores, false_scores)
    plot_true_false_score_historgrams(true_scores, false_scores, exclude_threshold={'True': 1,
                                                                                    'False': 0})


def main():
    models_to_compare = ['Original',
                         'Perfect',
                         'CalcProbabilityModel_g44_d50_b2_dw64',
                         'CalcProbabilityModel_g44_d45_b2b2_f15',
                         'CalcProbabilityModel_g44_d35_b2b2_f15']
    #plot_probability_histogram(models_to_compare[1])
    #test_compare_diff_scores(model_names=models_to_compare, indexes=[1], flatten_params=['','','',''])
    combine_all_diff_scores(model_names=models_to_compare, use_log_diff=True, flatten_params=['', '', '', '', '', ''])
    save_diff_matrix_by_models(models_to_compare[2:])


if __name__ == '__main__':
    main()