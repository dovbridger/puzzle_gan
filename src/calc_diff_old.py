
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from utils.plot_utils import plot_histograms
from puzzle.java_utils import create_diff_matrix3d_with_model_evaluations, parse_java_scores

NUM_LOSS_DIGITS = 3

def create_diff_matrix_for_puzzle(puzzle_name, opt):

    print("doint puzzle " + puzzle_name)
    opt.phase = 'name={0}-part_size=64-orientation=h'.format(puzzle_name)
    data_loader = CreateDataLoader(opt)
    dataset_h = data_loader.load_data()
    opt.phase = 'name={0}-part_size=64-orientation=v'.format(puzzle_name)
    data_loader = CreateDataLoader(opt)
    dataset_v = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    for i, data in enumerate(dataset_h):
        model.set_input(data)
        model.predict_and_store()
    print('finished horiznotal')
    for i, data in enumerate(dataset_v):
        model.set_input(data)
        model.predict_and_store()
    print('finished vertical')
    create_diff_matrix3d_with_model_evaluations(puzzle_name, part_size=64,
                                                model=model, test_folder=opt.dataroot,
                                                model_name=model.name() + '_' + opt.name)


def compare_puzzle_scores(score_file):
    scores = parse_java_scores(score_file)
    vals = []
    for pairs in scores.values():
        for pair in pairs:
            vals.append(pair)
    direct = [x for x,y in vals if y == 'direct']
    none = [x for x,y in vals if y == 'none']
    plot_histograms((direct, none), num_bins=10, labels=['New', 'Original'])


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