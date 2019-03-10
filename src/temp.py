from puzzle.scoring_statistics import run_diff_score_comparison, combine_all_diff_scores,\
    make_true_false_score_histogram, plot_true_false_score_historgrams
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
                                      burn_extent='4')

def plot_probability_histogram(model_name, indexes=range(1, 21)):
    true_scores, false_scores = [], []
    import numpy as np
    from puzzle.java_utils import get_probability_matrix_file_name
    for puzzle_name in [str(x) + 'b' for x in indexes]:
        file_name = get_probability_matrix_file_name(puzzle_name, model_name)
        with open(file_name, 'r') as f:
            probability_matrix = np.array(json.load(f))
        make_true_false_score_histogram(probability_matrix, 15, 10, true_scores, false_scores)
    plot_true_false_score_historgrams(true_scores, false_scores, exclude_threshold={'True': 0.8,
                                                                                    'False': 0.2})


def main():
    models_to_compare = ['Original',
                         'CalcProbabilityModel_g40_d40_b4_v']
    plot_probability_histogram(models_to_compare[1])
    #test_compare_diff_scores(model_names=models_to_compare, flatten_params=['','',(0.7,10),(0.51,10)])
    #combine_all_diff_scores(model_names=models_to_compare, use_log_diff=True, flatten_params=['','','','',''])
    #save_diff_matrix_by_models(models_to_compare[2:3])


if __name__ == '__main__':
    main()