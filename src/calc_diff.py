
from options.test_options import TestOptions
from data import CreateDataLoader
from argparse import Namespace
from data.virtual_puzzle_dataset import VirtualPuzzleDataset
from models import create_model
from utils.plot_utils import plot_histograms
from puzzle.java_utils import create_diff_matrix3d_with_model_evaluations, parse_java_scores

NUM_LOSS_DIGITS = 3

def create_diff_matrix_for_puzzle(puzzle_name, opt):

    print("doint puzzle " + puzzle_name)
    opt.puzzle_name = puzzle_name
    dataset = VirtualPuzzleDataset()
    dataset.initialize(opt)
    model = create_model(opt)
    model.setup(opt)
    create_diff_matrix3d_with_model_evaluations(opt.puzzle_name, opt.part_size, model, dataset)


def compare_puzzle_scores(score_file):
    scores = parse_java_scores(score_file)
    vals = []
    for pairs in scores.values():
        for pair in pairs:
            vals.append(pair)
    direct = [x for x,y in vals if y == 'direct']
    none = [x for x,y in vals if y == 'none']
    print(scores)
    print("New Average: {0}".format(sum(direct) / len(direct)))
    print("Old Average: {0}".format(sum(none) / len(none)))
    plot_histograms((direct, none), num_bins=10, labels=['New', 'Original'])


def calc_diff():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1
    for puzzle_name in [str(x) + 'b' for x in range(1, 2)]:
        create_diff_matrix_for_puzzle(puzzle_name, opt)

def main():
    calc_diff()


if __name__ == '__main__':
    main()