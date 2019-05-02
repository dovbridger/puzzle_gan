from .base_model import BaseModel
from . import networks
from .puzzle_gan_model import PuzzleGanModel
import torch
from utils.network_utils import get_network_file_name
import json
import os.path
from data.puzzle_with_old_inpainting_dataset import OLD_INPAINTING_NAME

FAKE_NAME = 'fake'

class TestCompareModel(BaseModel):
    def name(self):
        return 'TestCompareModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = PuzzleGanModel.modify_commandline_options(parser, is_train)
       # parser.set_defaults(dataset_name='puzzle_with_old_inpainting')
        parser.add_argument('--generators', type=str, default='[["puzzle_example_burn2", "latest"],["puzzle_example_burn2_b1", "latest"]]',
                            help="Json string representing a list, in which each item is a 2 element list of strings"
                            "as such: ['<generator name>','<epoch to load>']")
        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        opt.generators = opt.generators.replace(chr(39), chr(34))
        opt.generators = json.loads(opt.generators)
        self.nets = {}
        self.generator_names = []
        for name, epoch in opt.generators:
            generator_name = name + "_e" + epoch
            current_net = networks.get_generator(opt)
            self.load_network(os.path.join(opt.checkpoints_dir, name, get_network_file_name(epoch, 'G')), current_net)
            self.nets[generator_name] = current_net
            self.generator_names.append(generator_name)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.visual_names = ['burnt', 'real']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names += [FAKE_NAME + '_' + generator for generator in self.generator_names]

        if opt.dataset_name == 'puzzle_with_old_inpainting':
            self.visual_names.append(FAKE_NAME + '_' + OLD_INPAINTING_NAME)
        self.loss_names = [name for name in self.visual_names if name not in ['burnt', 'real']]
        self.model_names = []
        self.L1_loss = torch.nn.L1Loss()


    def set_input(self, input):
        self.real = input['real'].to(self.device)
        self.burnt = input['burnt'].to(self.device)
        self.image_paths = input['name']
        if self.opt.dataset_name == 'puzzle_with_old_inpainting':
            self.fake_old_inpainting = input[OLD_INPAINTING_NAME].to(self.device)

    def forward(self):
        real_in_generated_window = None
        for generator_name in self.generator_names:
            net = self.nets[generator_name]
            # fist iteration
            if real_in_generated_window is None:
                # Create ground truth window for loss calulation in ann models
                real_in_generated_window = self.real[:, :, self.opt.burn_extent: -self.opt.burn_extent,
                                                     net.generated_columns_start:net.generated_columns_end]
                if hasattr(self, 'fake_old_inpainting'):
                # Create the fake window for the old inpainting
                    fake_in_generated_window = self.fake_old_inpainting[:, :, self.opt.burn_extent: -self.opt.burn_extent,
                                                                        net.generated_columns_start:net.generated_columns_end]
                    # Calculate old inpainting loss
                    setattr(self, 'loss_' + FAKE_NAME + "_" + OLD_INPAINTING_NAME,
                            self.L1_loss(fake_in_generated_window, real_in_generated_window))
            current_fake = net(self.burnt)
            image_attribute_name = FAKE_NAME + "_" + generator_name
            setattr(self, image_attribute_name, current_fake)
            fake_in_generated_window = current_fake[:, :, self.opt.burn_extent: -self.opt.burn_extent,
                                                    net.generated_columns_start:net.generated_columns_end]
            current_loss = self.L1_loss(fake_in_generated_window, real_in_generated_window)
            setattr(self, 'loss_' + image_attribute_name, current_loss)

