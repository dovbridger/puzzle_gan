
import models.networks as networks
from models.base_model import BaseModel
import os
from utils.network_utils import get_network_file_name, get_discriminator_input
from torch import Tensor, unsqueeze

CALC_DIFF_MODEL_NAME = 'CalcDiffModel'

class CalcDiffModel(BaseModel):

    def name(self):
        return CALC_DIFF_MODEL_NAME

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(dataset_name='puzzle')
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(model_suffix='_post')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert opt.generator_window >= 2 * opt.burn_extent,\
            "The generator window({0}) is not large enough to inpaint the burnt area({1})".format(opt.generator_window, 2 * opt.burn_extent)
        assert not opt.isTrain, "calc_diff model is not to be used in training mode"
        self.netG = networks.get_generator(opt)
        generator_network_path = os.path.join(opt.checkpoints_dir,
                                              opt.network_to_load,
                                              get_network_file_name('latest', 'G'))
        self.load_network(generator_network_path, self.netG)

        self.netD = networks.get_discriminator(opt)
        discriminator_network_path = os.path.join(opt.checkpoints_dir,
                                                  opt.network_to_load,
                                                  get_network_file_name(opt.network_load_epoch, 'D' + opt.model_suffix))
        self.load_network(discriminator_network_path, self.netD)
        self.all_predictions = {}
        self.visual_names = ['fake']

    def set_input(self, input):
        self.burnt = input['burnt'].to(self.device)
        self.image_paths = input['name']

    def forward(self):
        self.fake = self.netG(self.burnt)
        self.prediction = self.netD(get_discriminator_input(self.opt, self.burnt, self.fake))

    def predict_and_store(self):
        self.test()
        reshaped_predictions = self.get_prediction()
        for i in range(len(self.image_paths)):
            self.all_predictions[self.image_paths[i]] = reshaped_predictions[i].item()

    def get_prediction(self):
        num_examples = self.prediction.shape[0]
        return self.prediction.reshape(num_examples, -1).mean(dim=1)

    def predict(self, data_example):
        for key in data_example:
            if isinstance(data_example[key], Tensor):
                data_example[key] = unsqueeze(data_example[key], 0)
        self.set_input(data_example)
        self.test()
        return self.prediction.mean().item()

    def instance_name(self):
        return self.name() + '_' + self.opt.name








