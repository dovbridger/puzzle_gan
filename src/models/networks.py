import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
from utils.network_utils import get_generator_mask, get_centered_window_indexes


class UnetLeftBlock(nn.Module):
    '''
     An Encoder block that performs the required normalization, a convolution, and activation
    '''
    def __init__(self, input_nc, output_nc, innermost=False, norm_layer=nn.BatchNorm2d, kernel_size=4):
        '''

        :param input_nc: number of input channels
        :param output_nc: number of output channels
        :param innermost: boolean, is this the last encoder block of the network?
        :param norm_layer: The requested normalization layer
        '''
        super(UnetLeftBlock, self).__init__()
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # With these parameters the width and height of the output will be half of the input's
        self.downconv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1, bias=use_bias)
        if not innermost:
            self.downnorm = norm_layer(output_nc)
            self.downrelu = nn.LeakyReLU(0.2, True)
        else:
            self.downnorm = None
            self.downrelu = nn.ReLU(True)

    def forward(self, x):
        y = self.downconv(x)
        if self.innermost:
            return self.downrelu(y)
        y = self.downnorm(y)
        return self.downrelu(y)


class UnetRightBlock(nn.Module):
    '''
    A decoder block that performs the required normalization, a de-convolution, and activation
    '''
    def __init__(self, input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d, kernel_size=4):
        '''
        :param input_nc: number of input channels
        :param output_nc: number of output channels
        :param outermost: boolean, is this the last decoder block of the network?
        :param norm_layer: The requested normalization layer
        '''
        super(UnetRightBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # With these parameters the width and height of the output will be twice as the input's
        self.upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1, bias=use_bias)
        if not outermost:
            self.upnorm = norm_layer(output_nc)
            self.uprelu = nn.ReLU(True)
        else:
            self.upnorm = None
            self.uprelu = None
            self.tanh = nn.Tanh()

    def forward(self, x):
        y = self.upconv(x)
        if self.outermost:
            return self.tanh(y)
        y = self.upnorm(y)
        return self.uprelu(y)


class UnetGenerator(nn.Module):
    '''
    The network for the Generator. consists of:
    6 encoder layers
    no bottleneck
    6 decoder layers
    For example, a 64 x 128 x 3 image will be encoded to a 1 x 2 x (ngf * 8) at the end of the encoder and back to
    64 x 128 x 3 at the end of the decoder
    Skip connections between encoder and decoder layers of same size are implemented in the 'forward' method
    '''
    def __init__(self, input_nc, output_nc, generator_window, input_size, ngf=64, norm_layer=nn.BatchNorm2d,
                 kernel_size=4, burn_extent=0):
        super(UnetGenerator, self).__init__()
        self.has_extra_layer = input_size[1] >= 256
        self.generated_columns_start, self.generated_columns_end = get_centered_window_indexes(input_size[1], generator_window)
        self.generated_window_mask = get_generator_mask(input_size,
                                                        (self.generated_columns_start, self.generated_columns_end),
                                                        burn_extent)
        # 64 x 128 in -> 32 x 64 out
        self.layer0_d = UnetLeftBlock(input_nc, ngf, norm_layer=norm_layer, kernel_size=kernel_size)
        # 32 x 64 in -> 16 x 32 out
        self.layer1_d = UnetLeftBlock(ngf, ngf*2, norm_layer=norm_layer, kernel_size=kernel_size)
        # 16 x 32 in -> 8 x 16 out
        self.layer2_d = UnetLeftBlock(ngf*2, ngf*4, norm_layer=norm_layer, kernel_size=kernel_size)
        # 8 x 16 in -> 4 x 8 out
        self.layer3_d = UnetLeftBlock(ngf*4, ngf*8, norm_layer=norm_layer, kernel_size=kernel_size)
        # 4 x 8 in -> 2 x 4 out
        self.layer4_d = UnetLeftBlock(ngf*8, ngf*8, norm_layer=norm_layer, kernel_size=kernel_size)
        # 2 x 4 in -> 1 x 2 out
        if self.has_extra_layer:
            self.layer5_d = UnetLeftBlock(ngf*8, ngf*8, norm_layer=norm_layer, kernel_size=kernel_size)
            self.layer5_u = UnetRightBlock(ngf*8*2, ngf*8, norm_layer=norm_layer, kernel_size=kernel_size)

        self.middle_d = UnetLeftBlock(ngf*8, ngf*8, norm_layer=norm_layer, kernel_size=kernel_size, innermost=True)
        # 1 x 2 in -> 2 x 4 out
        self.middle_u = UnetRightBlock(ngf*8, ngf*8, norm_layer=norm_layer, kernel_size=kernel_size)
        # 2 x 4 in -> 4 x 8 out
        self.layer4_u = UnetRightBlock(ngf*8*2, ngf*8, norm_layer=norm_layer, kernel_size=kernel_size)
        # 4 x 8 in -> 8 x 16 out
        self.layer3_u = UnetRightBlock(ngf*8*2, ngf*4, norm_layer=norm_layer, kernel_size=kernel_size)
        # 8 x 16 in -> 16 x 32 out
        self.layer2_u = UnetRightBlock(ngf*4*2, ngf*2, norm_layer=norm_layer, kernel_size=kernel_size)
        # 16 x 32 in -> 32 x 64 out
        self.layer1_u = UnetRightBlock(ngf*2*2, ngf, norm_layer=norm_layer, kernel_size=kernel_size)
        # 32 x 64 in -> 64 x 128 out
        self.layer0_u = UnetRightBlock(ngf*2, output_nc, norm_layer=norm_layer, kernel_size=kernel_size, outermost=True)

        self.layer0_d_out = self.layer1_d_out = self.layer2_d_out = self.layer3_d_out =\
            self.layer4_d_out = self.middle_d_in = None
        self.middle_d_out = self.middle_u_out = None
        self.in_layer4_u = self.layer4_u_out = self.layer3_u_out =\
            self.layer2_u_out = self.layer1_u_out = self.layer0_u_out = None

    def forward(self, input):
        # layerx_d_out: The output from layer 'x' of the encoder
        self.layer0_d_out = self.layer0_d(input)
        self.layer1_d_out = self.layer1_d(self.layer0_d_out)
        self.layer2_d_out = self.layer2_d(self.layer1_d_out)
        self.layer3_d_out = self.layer3_d(self.layer2_d_out)
        self.layer4_d_out = self.layer4_d(self.layer3_d_out)

        if self.has_extra_layer:
            self.layer5_d_out = self.layer5_d(self.layer4_d_out)
            self.middle_d_in = self.layer5_d_out
        # middle_d is the last layer of the encoder and middle u is the first layer of the decoder that follows
        else:
            self.middle_d_in = self.layer4_d_out
        self.middle_d_out = self.middle_d(self.middle_d_in)
        self.middle_u_out = self.middle_u(self.middle_d_out)

        if self.has_extra_layer:
            in_5 = torch.cat([self.middle_u_out, self.layer5_d_out], 1)
            self.in_layer4_u = self.layer5_u(in_5)
        else:
            # Input to decoder layer 4, before concatenating skip connection
            self.in_layer4_u = self.middle_u_out


        # in_x is the input to layer 'x' of the decoder (counting from the end) after concatenating the skip connection
        # From the output of layer 'x' of the encoder
        in_4 = torch.cat([self.in_layer4_u, self.layer4_d_out], 1)
        self.layer4_u_out = self.layer4_u(in_4)
        in_3 = torch.cat([self.layer4_u_out, self.layer3_d_out], 1)
        self.layer3_u_out = self.layer3_u(in_3)
        in_2 = torch.cat([self.layer3_u_out, self.layer2_d_out], 1)
        self.layer2_u_out = self.layer2_u(in_2)
        in_1 = torch.cat([self.layer2_u_out, self.layer1_d_out], 1)
        self.layer1_u_out = self.layer1_u(in_1)
        in_0 = torch.cat([self.layer1_u_out, self.layer0_d_out], 1)
        self.layer0_u_out = self.layer0_u(in_0)

        self.final_output = torch.where(self.generated_window_mask == 1, self.layer0_u_out, input)
        return self.final_output

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


# This Discriminator is not used in Dov's code
class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class NLayerDiscriminator(nn.Module):
    '''
    The structure is similar to the first half of the generator network ('UnetGenerator')
    '''
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if isinstance(target_is_real, bool):
            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            return target_tensor.expand_as(input)
        else:
            return target_is_real.view(-1, 1, 1, 1).expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)



def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # if len(gpu_ids) > 0:
    #    assert(torch.cuda.is_available())
    #    net.to(gpu_ids[0])
    #    net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    net = net.cuda()
    return net

def get_generator(opt):
    netG = UnetGenerator(opt.input_nc, opt.output_nc, opt.generator_window, opt.fineSize,
                         ngf=opt.ngf, norm_layer=get_norm_layer(opt.norm), kernel_size=opt.kernel_size,
                         burn_extent=opt.burn_extent)
    return init_net(netG, opt.init_type, opt.init_gain, opt.gpu_ids)


def get_discriminator(opt):
    discriminator_input_nc = opt.output_nc
    n_layers = 4 if opt.fineSize[1] >= 256 else 3
    netD = NLayerDiscriminator(discriminator_input_nc, opt.ndf,
                               n_layers=n_layers, norm_layer=get_norm_layer(opt.norm), use_sigmoid=opt.no_lsgan)
    return init_net(netD, opt.init_type, opt.init_gain, opt.gpu_ids)
