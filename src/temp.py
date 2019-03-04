from puzzle.scoring_statistics import run_diff_score_comparison, combine_all_diff_scores


def test_compare_diff_scores(models_to_compare, indexes=range(1, 21), correction_method='none', use_log_diff=False):
    for puzzle_name in [str(x) + 'b' for x in indexes]:
        print("doing puzzle " + puzzle_name)
        run_diff_score_comparison(puzzle_name,
                                  correction_method=correction_method,
                                  model_names=models_to_compare,
                                  use_log_diff=use_log_diff)


def main():
    models_to_compare = ['Original',
                         'Perfect',
                         'CalcDiffModel_g44_d30_b4_v',
                         'CalcDiffModel_g44_d30_b4_vp']

    test_compare_diff_scores(models_to_compare, [1,2,3], use_log_diff=True)
    #combine_all_diff_scores(model_names=models_to_compare, indexes=[1,2,3,4,5])
    #from puzzle.java_utils import save_diff_matrix_cnn_for_java
    #save_diff_matrix_cnn_for_java([str(x) + 'b' for x in range(1, 2)], 'CalcDiffModel_g44_d40_b4_virtual4', correction_method='direct', burn_extent='4')

if __name__ == '__main__':
    main()