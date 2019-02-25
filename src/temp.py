from puzzle.scoring_statistics import run_diff_score_comparison, combine_all_diff_scores


def test_compare_diff_scores(models_to_compare, indexes=range(1, 21)):
    for puzzle_name in [str(x) + 'b' for x in indexes]:
        print("doing puzzle " + puzzle_name)
        run_diff_score_comparison(puzzle_name,
                                  correction_method='direct',
                                  model_names=models_to_compare)


def main():
    models_to_compare = ['Original',
                         'Perfect',
                         'CalcDiffModel_g36_d30_b4',
                         'CalcDiffModel_g36_d40_b4',
                         'CalcDiffModel_g44_d40_b4']
    #test_compare_diff_scores(models_to_compare, [7])
    #combine_all_diff_scores(model_names=models_to_compare, correction_method='direct')
    from puzzle.java_utils import save_diff_matrix_cnn_for_java
    save_diff_matrix_cnn_for_java([str(x) + 'b' for x in range(1, 21)], 'CalcDiffModel_g36_d40_b4', correction_method='direct', burn_extent='4')

if __name__ == '__main__':
    main()