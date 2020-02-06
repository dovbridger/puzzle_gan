
from options.test_options import TestOptions
from data.virtual_puzzle_dataset import VirtualPuzzleDataset
from models import create_model
from utils.plot_utils import plot_histograms
from puzzle.puzzle_utils import get_info_from_file_name
from puzzle.java_utils import create_probability_matrix3d_with_model_evaluations, parse_java_scores,\
    save_diff_matrix_cnn_for_java

from os import path, listdir
from globals import NAME_MAGIC, BURN_EXTENT
import time

def create_diff_matrix_for_puzzle(puzzle_name, opt):
    print("doint puzzle " + puzzle_name)
    opt.puzzle_name = puzzle_name
    dataset = VirtualPuzzleDataset()
    dataset.initialize(opt)
    model = create_model(opt)
    model.setup(opt)
    start_time = time.time()
    create_probability_matrix3d_with_model_evaluations(opt.puzzle_name, opt.part_size, model, dataset)
    duration = time.time() - start_time
    print("Time to complete {0} with batch {1} was {2}".format(opt.puzzle_name, opt.batchSize, duration))
    print("Saving diff file for java")
    save_diff_matrix_cnn_for_java([puzzle_name], model.name() + "_" + opt.name,
                                  part_size=opt.part_size,
                                  burn_extent=opt.burn_extent)
    print("diff file created successfully")


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
    return results


def calc_diff():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    #opt.batchSize = 1
    print("sleeping %s seconds" % opt.delay_start)
    time.sleep(opt.delay_start)
    print("starting")
    test_files = listdir(path.join(opt.dataroot, 'test'))
    puzzle_names = [get_info_from_file_name(file, NAME_MAGIC) for file in test_files if file.endswith('.jpg') or file.endswith('.png')]
    for puzzle_name in puzzle_names:
        create_diff_matrix_for_puzzle(puzzle_name, opt)

def save_diff_matrix_by_models(model_names, puzzle_names, burn_extent=BURN_EXTENT):
    for model_name in model_names:
        save_diff_matrix_cnn_for_java(puzzle_names, model_name,
                                      burn_extent=burn_extent)
def main():
    calc_diff()


if __name__ == '__main__':
    main()
