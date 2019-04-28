from .base_model import BaseModel
from . import networks
from .puzzle_gan_model import PuzzleGanModel
import torch
from utils.network_utils import get_discriminator_input, get_network_file_name
from os import path, system


class PostGanVirtualModel(BaseModel):
    '''
    This model is used to post train the discriminator to distinguish between true neighbors and false neighbors
    The generator that is used here is loaded from the a previous GAN training process and does not change within this
    model.
    '''
    def name(self):
        return 'PostGanVirtualModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = PuzzleGanModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--model_suffix', type=str, default='_post',
                            help='In checkpoints_dir, [which_epoch]_net_D[model_suffix].pth will'
                            ' be loaded as the discriminator of PostGanModel')

        # Real images loss weight will be 1 - fake_loss_weight
        parser.add_argument('--fake_loss_weight', type=float,  default=0.5,
                            help="Weight of the loss on fake images in the total loss calculation")
        parser.set_defaults(dataset_name='virtual_puzzle')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['burnt', 'fake']
        if self.opt.fake_loss_weight < 1:
            self.visual_names.append('real')

        if self.opt.coupled_false:
            self.visual_names += ['false_' + name for name in self.visual_names]
            self.false_label = torch.zeros((self.opt.batchSize), dtype=torch.float32).to(self.device)
        self.model_names = ['D' + opt.model_suffix]
        self.netD = networks.get_discriminator(opt)
        setattr(self, 'netD' + opt.model_suffix, self.netD)
        self.netG = networks.get_generator(opt)
        post_train_discriminator_path = path.join(self.save_dir,
                                                  get_network_file_name(opt.which_epoch, 'D' + opt.model_suffix))
        post_train_generator_path = path.join(self.save_dir, get_network_file_name('latest', 'G'))
        if path.exists(post_train_discriminator_path):
            print("A post training discriminator already exists, a clean discriminator will not be copied")
        else:
            clean_discriminator_path = path.join(opt.checkpoints_dir,
                                                 opt.network_to_load,
                                                 get_network_file_name(opt.network_load_epoch, 'D'))
            clean_discriminator_target_path = path.join(self.save_dir,
                                                        get_network_file_name('latest', 'D' + opt.model_suffix))
            print("copying clean discriminator from '{0}' to '{1}'".format(
                clean_discriminator_path, clean_discriminator_target_path))
            system('copy  "{0}" "{1}"'.format(clean_discriminator_path, clean_discriminator_target_path))
            self.load_network(clean_discriminator_target_path, self.netD)

            # Use the same GAN setup (network name + epoch)
            # of the discriminator for the generator as well
            source_generator_network_path = path.join(opt.checkpoints_dir,
                                               opt.network_to_load,
                                               get_network_file_name(opt.network_load_epoch, 'G'))
            system('copy  "{0}" "{1}"'.format(source_generator_network_path, post_train_generator_path))
#        print("Discriminator will not be loaded, starting from scratch")
        self.load_network(post_train_generator_path, self.netG)
        print("Pretrainted discriminator loaded")
        self.dataset_access = None

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_D]
            # specify the training losses you want to print out. The program will call base_model.get_current_losses
            self.loss_names = ['D', 'D_fake']
            if self.opt.fake_loss_weight < 1:
                self.loss_names.append('D_real')

    def set_input(self, input):
        for visual_name in self.visual_names:
            if 'fake' not in visual_name:
                setattr(self, visual_name, input[visual_name].to(self.device))

        # Is also used in the loss calculation so needs to be on gpu
        self.label = input['label'].float().to(self.device)
        self.image_paths = input['name']

    def forward(self):
        self.fake = self.netG(self.burnt)

        if not self.opt.isTrain:
            self.probability = self.netD(get_discriminator_input(self.opt, self.fake))

    def backward(self):
        # Real
        if self.opt.fake_loss_weight < 1:
            discriminator_real_input = get_discriminator_input(self.opt, self.real)
            prediction_real = self.netD(discriminator_real_input)
            self.loss_D_real = self.criterionGAN(prediction_real, self.label)

        # Fake
        discriminator_fake_input = get_discriminator_input(self.opt, self.fake)
        # stop backprop to the generator by detaching 'discriminator_fake_input'
        prediction_fake = self.netD(discriminator_fake_input.detach())
        self.loss_D_fake = self.criterionGAN(prediction_fake, self.label)

        if self.opt.coupled_false:
            # False neighbors
            # Real
            if self.opt.fake_loss_weight < 1:
                discriminator_false_real_input = get_discriminator_input(self.opt, self.false_real)
                prediction_false_real = self.netD(discriminator_false_real_input)
                self.loss_D_real += self.criterionGAN(prediction_false_real, self.false_label)

            # Fake
            discriminator_false_fake_input = get_discriminator_input(self.opt, self.false_fake)
            # stop backprop to the generator by detaching 'discriminator_false_fake_input'
            prediction_false_fake = self.netD(discriminator_false_fake_input.detach())
            self.loss_D_fake += self.criterionGAN(prediction_false_fake, self.false_label)
        # Combined loss
        if self.opt.fake_loss_weight < 1:
            self.loss_D = self.loss_D_real * (1 - self.opt.fake_loss_weight) + self.loss_D_fake * self.opt.fake_loss_weight
        else:
            self.loss_D = self.loss_D_fake
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward()
        self.optimizer_D.step()

    def get_prediction(self):
        num_examples = self.probability.shape[0]
        return self.probability.reshape(num_examples, -1).mean(dim=1), self.label


