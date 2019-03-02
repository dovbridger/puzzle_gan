
import models.networks as networks
from models.base_model import BaseModel
from utils.network_utils import get_generator_path


class InpaintingModel(BaseModel):
    def name(self):
        return 'InpaintingModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(dataset_name='puzzle')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert not opt.isTrain, "Inpainting model is not to be used in training mode"
        self.model_names = []
        self.visual_names = ['first', 'second', 'correct']
        self.netG = networks.get_generator(opt)
        self.load_network(get_generator_path(opt), self.netG)

    def set_input(self, input):
        self.burnt = input['burnt'][0].to(self.device)
        self.image_paths = input['name']

    def forward(self):
        for i, visual in enumerate(self.visual_names):
            if i < self.burnt.shape[0]:
                setattr(self, visual, self.netG(self.burnt[i].unsqueeze(0)))
            else:
                setattr(self, visual, None)



