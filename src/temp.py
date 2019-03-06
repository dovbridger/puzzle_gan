from puzzle.scoring_statistics import run_diff_score_comparison, combine_all_diff_scores


def test_compare_diff_scores(models_to_compare, indexes=range(1, 21), correction_method='none'):
    for puzzle_name in [str(x) + 'b' for x in indexes]:
        print("doing puzzle " + puzzle_name)
        run_diff_score_comparison(puzzle_name,
                                  correction_method=correction_method,
                                  model_names=models_to_compare,
                                  use_log_diff=False,
                                  flatten_params=None)

def save_diff_matrix_by_models(model_names):
    from puzzle.java_utils import save_diff_matrix_cnn_for_java
    for model_name in model_names:
        save_diff_matrix_cnn_for_java([str(x) + 'b' for x in range(1, 21)], model_name,
                                      additional_params='0',
                                      burn_extent='4')

def main():
    models_to_compare = ['Original',
                         'Perfect',
                         'CalcProbabilityModel_g36_d30_b4_v',
                         'CalcProbabilityModel_g44_d40_b4_v']
    #test_compare_diff_scores(models_to_compare, [1,2,3], use_log_diff=True)
    #combine_all_diff_scores(model_names=models_to_compare, use_log_diff=True, flatten_params=(0.7, 20))
    save_diff_matrix_by_models(models_to_compare[2:3])


if __name__ == '__main__':
    main()