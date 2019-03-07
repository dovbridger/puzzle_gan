from puzzle.scoring_statistics import run_diff_score_comparison, combine_all_diff_scores


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

def main():
    models_to_compare = ['Original',
                         'CalcProbabilityModel_g40_d30_b4_v']
    #test_compare_diff_scores(model_names=models_to_compare,
                       #      flatten_params=['','',(0.7,10),(0.51,10)])
    combine_all_diff_scores(model_names=models_to_compare, use_log_diff=True,
                            flatten_params=['','','','',''])
    #save_diff_matrix_by_models(models_to_compare[2:3])


if __name__ == '__main__':
    main()