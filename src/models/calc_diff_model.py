
import models.networks as networks
from models.base_model import BaseModel
import os
from utils.network_utils import get_network_file_name, get_discriminator_input


class CalcDiffModel(BaseModel):
    def name(self):
        return 'CalcDiffModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(dataset_name='puzzle')
        parser.add_argument('--generator_to_load', type=str, default='batch1',
                            help='Which generator model to load')
        parser.add_argument('--generator_epoch', type=str, default='latest')
        parser.add_argument('--discriminator_to_load', type=str, default='batch1_post',
                            help='Which discriminator model to load')
        parser.add_argument('--discriminator_epoch', type=str, default='latest')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert opt.generator_window >= 2 * opt.burn_extent,\
            "The generator window({0}) is not large enough to inpaint the burnt area({1})".format(opt.generator_window, 2 * opt.burn_extent)
        assert not opt.isTrain, "calc_diff model is not to be used in training mode"
        self.netG = networks.get_generator(opt)
        generator_network_path = os.path.join(opt.checkpoints_dir,
                                              opt.generator_to_load,
                                              get_network_file_name(opt.generator_epoch, 'G'))
        self.load_network(generator_network_path, self.netG)

        self.netD = networks.get_discriminator(opt)
        discriminator_network_path = os.path.join(opt.checkpoints_dir,
                                                  opt.discriminator_to_load,
                                                  get_network_file_name(opt.discriminator_epoch, 'D'))
        self.load_network(discriminator_network_path, self.netD)

    def set_input(self, input):
        self.burnt = input['burnt'].to(self.device)

    def forward(self):
        fake = self.netG(self.burnt)
        self.prediction = self.netD(get_discriminator_input(self.opt, self.burnt, fake))

    def predict(self):
        self.test()
        return self.prediction.mean().item()


