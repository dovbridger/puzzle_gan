from .base_model import BaseModel
from . import networks
from .puzzle_gan_model import PuzzleGanModel
import torch
from utils.network_utils import get_discriminator_input, get_network_file_name
import json

class TestCompareModel(BaseModel):
    def name(self):
        return 'TestCompareModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = PuzzleGanModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_name='puzzle')
        parser.add_argument('--generators', type=str, default='[["puzzle_example_burn2", "latest"],["puzzle_example_burn2_b1", "latest"]]')
        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        opt.generators = json.loads(opt.generators)
        self.nets = {}
        self.generator_names = []
        for name, epoch in opt.generators:
            generator_name = name + "_epoch_" + epoch
            current_net = networks.get_generator(opt)
            current_net.load_network(os.path.join(opt.checkpoints_dir, get_network_file_name(epoch, name)))
            nets[generator_name] = current_net
            self.generator_names.append(generator_name)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = [generator +'_L1' for generator in self.generator_names]
        self.loss_names.append('old_inpainting')
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['fake_' + generator for generator in self.generator_names]
        self.visual_names.append('real')
        self.visual_names.append('burnt')
        self.visual_names.append('fake_old_inpainting')
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = []
        for generator, epoch in op:
        self.netG = networks.get_generator(opt)





    def set_input(self, input):
        self.real = input['real'].to(self.device)
        self.burnt = input['burnt'].to(self.device)
        self.image_paths = input['path']
        self.fake_old_inpainting = input['old_inpainting']

    def forward(self):
        real_in_generated_window = None
        for generator_name in self.generator_names:
            net = self.nets[generator_name]
            # fist iteration
            if real_in_generated_window is None:
                real_in_generated_window = self.real[:, :, :,
                                           net.generated_columns_start:net.generated_columns_end]
            current_fake = net(self.burnt)
            setattr(self, 'fake_' + generator_name, current_fake)
            fake_in_generated_window = current_fake[:, :, :,
                                       net.generated_columns_start:net.generated_columns_end]
            current_loss = torch.nn.L1Loss((fake_in_generated_window, real_in_generated_window))
            setattr(self,'loss_' + generator_name + "_L1", current_loss)




