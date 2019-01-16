from .base_model import BaseModel
from . import networks
from .puzzle_gan_model import PuzzleGanModel
import torch
from utils.network_utils import get_discriminator_input


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = PuzzleGanModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')
        parser.set_defaults(dataset_name='puzzle_test')
        parser.add_argument('--discriminator_test', action='store_true',
                            help='Whether or not to include the discriminator in the test to measure true/false neighbor'
                                 'identification')
        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_fake_true', 'D_fake_false']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['burnt_true', 'real_true', 'fake_true']
        for i in range(opt.num_false_examples):
            self.visual_names.append('burnt_false_' + str(i))
            self.visual_names.append('real_false_' + str(i))
            self.visual_names.append('fake_false_' + str(i))

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]

        self.netG = networks.get_generator(opt)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

        if opt.discriminator_test:
            self.netD = networks.get_discriminator(opt)
            setattr(self, 'netD' + opt.model_suffix, self.netD)
            # define loss function
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)

    def set_input(self, input):
        self.real_true = input['real_true'].to(self.device)
        self.burnt_true = input['burnt_true'].to(self.device)
        real_false_images = input['real_false']
        burnt_false_images = input['burnt_false']
        for i in range(self.opt.num_false_examples):
            if i < len(burnt_false_images):
                setattr(self, 'burnt_false_' + str(i), burnt_false_images[i].to(self.device))
                setattr(self, 'real_false_' + str(i), real_false_images[i].to(self.device))
            else:
                # If there aren't enough false examples set the attributes to None.
                # Not setting the attributes at all could lead to bugs later on in the visualizer
                setattr(self, 'burnt_false_' + str(i), None)
                setattr(self, 'real_false_' + str(i), None)

        self.image_paths = input['path']

    def forward(self):
        self.fake_true = self.netG(self.burnt_true)
        self.true_probability = self.netD(get_discriminator_input(self.opt, self.burnt_true, self.fake_true))
        self.false_probability = []
        # Create as many fake images as there exist false neighbor examples
        for i in range(self.opt.num_false_examples):
            burnt_false = getattr(self, 'burnt_false_' + str(i))
            if burnt_false is not None:
                fake_false = self.netG(burnt_false)
                self.false_probability.append(self.netD(get_discriminator_input(self.opt, burnt_false, fake_false)))
                setattr(self, 'fake_false_' + str(i), fake_false)
            else:
                setattr(self, 'fake_false_' + str(i), None)

    # Note - Currently only takes the first false probability in the list
    def get_probabilities(self):
        if len(self.false_probability) == 0:
            false_probability = torch.zeros(self.true_probability.shape)
        else:
            false_probability = self.false_probability[0]
        return {'True': self.true_probability, 'False': false_probability}

