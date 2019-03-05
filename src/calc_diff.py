
from options.test_options import TestOptions
from data.virtual_puzzle_dataset import VirtualPuzzleDataset
from models import create_model
from utils.plot_utils import plot_histograms
from puzzle.java_utils import create_probability_matrix3d_with_model_evaluations, parse_java_scores

def create_diff_matrix_for_puzzle(puzzle_name, opt):
    print("doint puzzle " + puzzle_name)
    opt.puzzle_name = puzzle_name
    dataset = VirtualPuzzleDataset()
    dataset.initialize(opt)
    model = create_model(opt)
    model.setup(opt)
    create_probability_matrix3d_with_model_evaluations(opt.puzzle_name, opt.part_size, model, dataset)


def compare_puzzle_scores(score_file):
    scores = parse_java_scores(score_file)
    vals = []
    for pairs in scores.values():
        for pair in pairs:
            vals.append(pair)
    method_names = list(set([y for x, y in vals]))
    results = tuple([x for x, y in vals if y == name] for name in method_names)
    for key in scores:
        print("{0}: {1}".format(key, scores[key]))
    for i in range(len(method_names)):
        print("{0}: Average={1}, max={2}, min={3}".format(method_names[i],
                                                          sum(results[i]) / len(results[i]),
                                                          max(results[i]),
                                                          min(results[i])))
    title = 'Puzzle Score Histogram Comparison'
    plot_histograms(results, num_bins=10, labels=method_names, titles=[title])


def calc_diff():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1
    for puzzle_name in [str(x) + 'b' for x in range(1, 21)]:
        create_diff_matrix_for_puzzle(puzzle_name, opt)


def main():
    calc_diff()


if __name__ == '__main__':
    main()
