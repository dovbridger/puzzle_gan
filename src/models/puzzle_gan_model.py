import torch
from utils.image_pool import ImagePool
from utils.network_utils import  get_discriminator_input
import models.networks as networks
from models.base_model import BaseModel


class PuzzleGanModel(BaseModel):
    def name(self):
        return 'PuzzleGanModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert opt.generator_window >= 2 * opt.burn_extent,\
            "The generator window({0}) is not large enough to inpaint the burnt area({1})".format(opt.generator_window, 2 * opt.burn_extent)
        self.isTrain = opt.isTrain

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        # self.loss_<item> must exist for every <item> in self.loss_names
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # self.<item> must exist for every <item> in self.visual_names
        self.visual_names = ['burnt', 'real', 'fake']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        self.netG = networks.get_generator(opt)
        if self.isTrain:
            self.netD = networks.get_discriminator(opt)
            self.burnt_and_fake_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr * opt.dlr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        self.real = input['real'].to(self.device)
        self.burnt = input['burnt'].to(self.device)
        self.image_paths = input['name']
        assert all(input['label'].tolist()), "PuzzleGanModel got a 'False (0)' label, something is wrong"

    def forward(self):
        self.fake = self.netG(self.burnt)

    def backward_D(self):
        discriminator_fake_input = self.burnt_and_fake_pool.query(get_discriminator_input(self.opt, self.burnt, self.fake))
        # stop backprop to the generator by detaching 'discriminator_fake_input'
        pred_fake = self.netD(discriminator_fake_input.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        discriminator_real_input = get_discriminator_input(self.opt, self.burnt, self.real)

        pred_real = self.netD(discriminator_real_input)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(burnt) should fool the discriminator
        discriminator_fake_input = get_discriminator_input(self.opt, self.burnt, self.fake)
        pred_fake = self.netD(discriminator_fake_input)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(burnt) = real
        fake_in_generated_window = self.fake[:, :, :, self.netG.generated_columns_start:self.netG.generated_columns_end]
        real_in_generated_window = self.real[:, :, :, self.netG.generated_columns_start:self.netG.generated_columns_end]
        self.loss_G_L1 = self.criterionL1(fake_in_generated_window, real_in_generated_window) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
