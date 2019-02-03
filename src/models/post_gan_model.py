from .base_model import BaseModel
from . import networks
from .puzzle_gan_model import PuzzleGanModel
import torch
from utils.network_utils import get_discriminator_input, get_network_file_name
from os import path, system


class PostGanModel(BaseModel):
    '''
    This model is used to post train the discriminator to distinguish between true neighbors and false neighbors
    The generator that is used here is loaded from the a previous GAN training process and does not change within this
    model.
    '''
    def name(self):
        return 'PostGanModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = PuzzleGanModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--model_suffix', type=str, default='_post',
                            help='In checkpoints_dir, [which_epoch]_net_D[model_suffix].pth will'
                            ' be loaded as the discriminator of PostGanModel')
        parser.add_argument('--generator_file_name', type=str, default='latest_net_G.pth',
                            help='File name of the generator model that we want to use to post train the discriminator')
        parser.add_argument('--copy_clean_discriminator', action='store_true',
                            help="If there is no pre-saved discriminator model from post training add this flag to copy"
                                 "the epoch specific model from the GAN phase as 'latest'")
        # Real images loss weight will be 1 - fake_loss_weight
        parser.add_argument('--fake_loss_weight', type=float,  default=0.5,
                            help="Weight of the loss on fake images in the total loss calculation")
        parser.set_defaults(dataset_name='puzzle_with_false')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['burnt_true', 'real_true', 'fake_true']
        for i in range(opt.num_false_examples):
            self.visual_names.append('burnt_false_' + str(i))
            self.visual_names.append('real_false_' + str(i))
            self.visual_names.append('fake_false_' + str(i))

        self.model_names = ['D' + opt.model_suffix]

        self.netD = networks.get_discriminator(opt)
        setattr(self, 'netD' + opt.model_suffix, self.netD)
        clean_discriminator_path = path.join(self.save_dir, get_network_file_name(opt.which_epoch, 'D'))
        post_train_discriminator_path = path.join(self.save_dir,
                                                  get_network_file_name(opt.which_epoch, 'D' + opt.model_suffix))
        if not path.exists(post_train_discriminator_path):
            print("No post train discriminator model file ('{0}')".format(post_train_discriminator_path))
            assert opt.copy_clean_discriminator,\
                "Can't copy clean discriminator because 'copy_clean_discriminator' is False"
            clean_discriminator_target_path = path.join(self.save_dir,
                                                        get_network_file_name('latest', 'D' + opt.model_suffix))
            print("copying clean discriminator from '{0}' to '{1}'".format(
                clean_discriminator_path, clean_discriminator_target_path))
            system('copy  "{0}" "{1}"'.format(clean_discriminator_path, clean_discriminator_target_path))

        # Defined separately and not included in 'self.model_names' because it is read only and is not being trained
        self.netG = networks.get_generator(opt)
        self.load_network(path.join(self.save_dir, opt.generator_file_name), self.netG)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_D]
            # specify the training losses you want to print out. The program will call base_model.get_current_losses
            self.loss_names = ['D', 'D_real_true', 'D_fake_true', 'D_real_false', 'D_fake_false']

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
        assert self.burnt_false_0 is not None, "No false neighbor exist for '{0}', All true neighbor images must " \
                                               "have at least 1 corresponding false neighbor".format(self.image_paths)
        self.fake_true = self.netG(self.burnt_true)
        # first false neighbor
        self.fake_false_0 = self.netG(self.burnt_false_0)

        if not self.opt.isTrain:
            self.true_probability = self.netD(get_discriminator_input(self.opt, self.burnt_true, self.fake_true))
            # false probablity pertains to the first false example only
            self.false_probability = self.netD(get_discriminator_input(self.opt, self.burnt_true, self.fake_false_0))

        # Create as many additional fake images as there exist additional false neighbor examples
        for i in range(1, self.opt.num_false_examples):
            burnt_false = getattr(self, 'burnt_false_' + str(i))
            if burnt_false is not None:
                fake_false = self.netG(burnt_false)
                setattr(self, 'fake_false_' + str(i), fake_false)
            else:
                setattr(self, 'fake_false_' + str(i), None)

    def backward(self):
        # True neighbors
        # Real
        discriminator_real_true_input = get_discriminator_input(self.opt, self.burnt_true, self.real_true)
        prediction_real_true = self.netD(discriminator_real_true_input)
        self.loss_D_real_true = self.criterionGAN(prediction_real_true, True)

        # Fake
        discriminator_fake_true_input = get_discriminator_input(self.opt, self.burnt_true, self.fake_true)
        # stop backprop to the generator by detaching 'discriminator_fake_true_input'
        prediction_fake_true = self.netD(discriminator_fake_true_input.detach())
        self.loss_D_fake_true = self.criterionGAN(prediction_fake_true, True)

        # False neighbors
        # Real
        discriminator_real_false_input = get_discriminator_input(self.opt, self.burnt_false_0, self.real_false_0)
        prediction_real_false = self.netD(discriminator_real_false_input)
        self.loss_D_real_false = self.criterionGAN(prediction_real_false, False)

        # Fake
        discriminator_fake_false_input = get_discriminator_input(self.opt, self.burnt_false_0, self.fake_false_0)
        # stop backprop to the generator by detaching 'discriminator_fake_false_input'
        prediction_fake_false = self.netD(discriminator_fake_false_input.detach())
        self.loss_D_fake_false = self.criterionGAN(prediction_fake_false, False)

        # Combined loss
        self.loss_D = (self.loss_D_real_true + self.loss_D_real_false) * (1 - self.opt.fake_loss_weight) +\
                      (self.loss_D_fake_true + self.loss_D_fake_false) * self.opt.fake_loss_weight
        self.loss_D.backward()


    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward()
        self.optimizer_D.step()

    def get_probabilities(self):
        return {'True': self.true_probability, 'False': self.false_probability}