from puzzle.scoring_statistics import run_diff_score_comparison, combine_all_diff_scores,\
    make_true_false_score_histogram, plot_true_false_score_historgrams, plot_histograms, get_figure_name
from puzzle.puzzle_utils import get_info_from_file_name, get_full_puzzle_name_from_characteristics, read_metadata
from globals import BURN_EXTENT, NAME_MAGIC, TEST_DATA_PATH, METADATA_FOLDER_NAME, HORIZONTAL, PART_SIZE
import json
from os import listdir, path
import numpy as np


def test_compare_diff_scores(model_names, indexes=range(1, 21), correction_method='none',
                             flatten_params=''):
    for puzzle_name in [str(x) + 'b' for x in indexes]:
        print("doing puzzle " + puzzle_name)
        run_diff_score_comparison(puzzle_name,
                                  correction_method=correction_method,
                                  model_names=model_names,
                                  use_log_diff=True,
                                  flatten_params=flatten_params)



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
def get_probability_scores(model_name, puzzle_names):
    true_scores, false_scores = [], []
    import numpy as np
    from puzzle.java_utils import get_probability_matrix_file_name
    for puzzle_name in puzzle_names:
        full_puzzle_name = get_full_puzzle_name_from_characteristics(puzzle_name,
                                                                     part_size=PART_SIZE,
                                                                     orientation=HORIZONTAL)
        metadata = read_metadata(path.join(TEST_DATA_PATH, METADATA_FOLDER_NAME), full_puzzle_name)
        file_name = get_probability_matrix_file_name(puzzle_name, model_name)
        num_x_parts = int(metadata['num_x_parts'])
        num_y_parts = int(metadata['num_y_parts'])
        with open(file_name, 'r') as f:
            probability_matrix = np.array(json.load(f))
        make_true_false_score_histogram(probability_matrix, num_x_parts, num_y_parts, true_scores, false_scores)
    return true_scores, false_scores

def compare_probability_scores(model_names, puzzle_names, model_titles=None):
    all_scores = []
    labels = model_names if model_titles is None else model_titles
    for i, model_name in enumerate(model_names):
        current_scores = get_probability_scores(model_name, puzzle_names)
        all_scores.append(current_scores)
        #plot_true_false_score_historgrams(current_scores[0], current_scores[1], description=labels[i])
    for label, index in [('Positive', 0), ('Negative', 1)]:
        title ='Discriminator Output Histogram on {0} Neighbor Examples'.format(label)
        data_to_plot = tuple(score[index] for score in all_scores)
        print('plotting {0} histograms'.format(label))
        for data in data_to_plot:
            print(np.array(data).mean())
        axes_labels = ['Discriminator Output', '# of Exampels']
        plot_histograms(data_to_plot, labels=labels, num_bins=10, titles=[title], linewidth=3, axes_labels=axes_labels,
                        output_file_name=get_figure_name("All", label + '-'.join(model_names)), loc='upper center')
        return data_to_plot


def main():
    PUZZLE_SUFFIX = 'c'
    test_files = listdir(TEST_DATA_PATH)
    puzzle_names = [get_info_from_file_name(file, NAME_MAGIC) for file in test_files if
                     PUZZLE_SUFFIX + '-' in file and file.endswith('.png')]
    print(puzzle_names)
    models_to_compare = ['Perfect',
                         'Original',
                         'CalcProbabilityModel_batch1_g48_d40_b2']
    compare_probability_scores(models_to_compare[2:], puzzle_names=puzzle_names, model_titles=['Our GAN Inpainting Discriminator', 'Fresh Discriminator', 'Fresh Discriminator(No Inpainting)'])
    #combine_all_diff_scores(model_names=models_to_compare[1:],
     #                       puzzle_names=puzzle_names, use_log_diff=True,
      #                      flatten_params=['', '', '', '', '', '',''],
       #                     model_titles=['original', 'd50','d40', 'd30'])

if __name__ == '__main__':
    main()