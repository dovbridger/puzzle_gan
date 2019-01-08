import torch
from utils.image_pool import ImagePool
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
        self.netG = networks.get_generator(opt.input_nc, opt.output_nc, opt.ngf, opt.init_type, opt.init_gain,
                                           self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.get_descriminator(opt.input_nc + opt.output_nc, opt.ndf, use_sigmoid, opt.init_type,
                                                   opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.burnt_and_fake_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real = input['real'].to(self.device)
        self.burnt = input['burnt'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        self.fake = self.netG(self.burnt)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching 'burnt_and_fake'
        burnt_and_fake = self.burnt_and_fake_pool.query(torch.cat((self.burnt, self.fake), 1))
        pred_fake = self.netD(burnt_and_fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        burnt_and_real = torch.cat((self.burnt, self.real), 1)
        pred_real = self.netD(burnt_and_real)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        burnt_cat_fake = torch.cat((self.burnt, self.fake), 1)
        pred_fake = self.netD(burnt_cat_fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake, self.real) * self.opt.lambda_L1

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
