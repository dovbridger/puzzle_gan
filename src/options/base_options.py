import argparse
import os
from utils import util
import torch
import models
import data
import os.path

class BaseOptions():
    '''
    General command line options for the code
    '''
    def __init__(self):
        self.initialized = False
        self.project_root = r'../'
        # Directory where models and results can be saved
        self.saved_data_root = os.path.join(self.project_root, 'saved_data')

    def initialize(self, parser):
        model = 'puzzle_gan'
        task = 'puzzle_try'
        experiment_name = 'no_burnt'
        loadSize = (64, 128)
        self.fine_size = (64, 128)
        data_root = os.path.join(self.project_root, 'datasets', 'puzzle_parts')
        batchSize = 64
        dataset_name = 'puzzle'
        parser.add_argument('--dataroot', type=str, default=data_root, help='path to images (should have subfolders train, validation, test)')
        parser.add_argument('--batchSize', type=int, default=batchSize, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=loadSize, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=self.fine_size, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default=(task + '_' + experiment_name), help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_name', type=str, default=dataset_name, help='name of the dataset to be loaded')
        parser.add_argument('--model', type=str, default=model, help='chooses which model to use')
        parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str,
                            default=os.path.join(self.saved_data_root, 'models'),
                            help='models are saved here')
        parser.add_argument('--norm', type=str, default='batch',
                            help='Type of normalization to use ("instance", "batch", "none")')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_port', type=int, default=1986, help='visdom port of the web display')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix')
        parser.add_argument('--display_ncols', type=int, default=1, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--kernel_size', type=int, default=4, help='Conv and Deconv kernel size')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_option_setter = data.get_option_setter(opt.dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
