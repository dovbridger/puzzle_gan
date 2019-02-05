
import models.networks as networks
from models.base_model import BaseModel


class InpaintingModel(BaseModel):
    def name(self):
        return 'InpaintingModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(dataset_name='puzzle')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert opt.generator_window >= 2 * opt.burn_extent,\
            "The generator window({0}) is not large enough to inpaint the burnt area({1})".format(opt.generator_window, 2 * opt.burn_extent)
        assert not opt.isTrain, "Inpainting model is not to be used in training mode"

        self.model_names = ['G']
        self.visual_names = ['fake']
        self.netG = networks.get_generator(opt)

    def set_input(self, input):
        self.burnt = input['burnt'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        self.fake = self.netG(self.burnt)
