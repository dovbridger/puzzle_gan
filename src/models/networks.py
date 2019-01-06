import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler


class UnetLeftBlock(nn.Module):
    def __init__(self, input_nc, output_nc, innermost=False, norm_layer=nn.BatchNorm2d):
        super(UnetLeftBlock, self).__init__()
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
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
    def __init__(self, input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
        super(UnetRightBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
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
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(UnetGenerator, self).__init__()
        # 64 x 128 in -> 32 x 64 out
        self.layer0_d = UnetLeftBlock(input_nc, ngf, norm_layer=norm_layer)
        # 32 x 64 in -> 16 x 32 out
        self.layer1_d = UnetLeftBlock(ngf, ngf*2, norm_layer=norm_layer)
        # 16 x 32 in -> 8 x 16 out
        self.layer2_d = UnetLeftBlock(ngf*2, ngf*4, norm_layer=norm_layer)
        # 8 x 16 in -> 4 x 8 out
        self.layer3_d = UnetLeftBlock(ngf*4, ngf*8, norm_layer=norm_layer)
        # 4 x 8 in -> 2 x 4 out
        self.layer4_d = UnetLeftBlock(ngf*8, ngf*8, norm_layer=norm_layer)
        # 2 x 4 in -> 1 x 2 out
        self.middle_d = UnetLeftBlock(ngf*8, ngf*8, norm_layer=norm_layer, innermost=True)
        # 1 x 2 in -> 2 x 4 out
        self.middle_u = UnetRightBlock(ngf*8, ngf*8, norm_layer=norm_layer)
        # 2 x 4 in -> 4 x 8 out
        self.layer4_u = UnetRightBlock(ngf*8*2, ngf*8, norm_layer=norm_layer)
        # 4 x 8 in -> 8 x 16 out
        self.layer3_u = UnetRightBlock(ngf*8*2, ngf*4, norm_layer=norm_layer)
        # 8 x 16 in -> 16 x 32 out
        self.layer2_u = UnetRightBlock(ngf*4*2, ngf*2, norm_layer=norm_layer)
        # 16 x 32 in -> 32 x 64 out
        self.layer1_u = UnetRightBlock(ngf*2*2, ngf, norm_layer=norm_layer)
        # 32 x 64 in -> 64 x 128 out
        self.layer0_u = UnetRightBlock(ngf*2, output_nc, norm_layer=norm_layer, outermost=True)

        self.layer0_d_out = self.layer1_d_out = self.layer2_d_out = self.layer3_d_out =\
            self.layer4_d_out = self.middle_d_in = None
        self.middle_d_out = self.middle_u_out = None
        self.in_layer4_u = self.layer4_u_out = self.layer3_u_out =\
            self.layer2_u_out = self.layer1_u_out = self.layer0_u_out = None

    def forward(self, x):
        self.layer0_d_out = self.layer0_d(x)
        self.layer1_d_out = self.layer1_d(self.layer0_d_out)
        self.layer2_d_out = self.layer2_d(self.layer1_d_out)
        self.layer3_d_out = self.layer3_d(self.layer2_d_out)
        self.layer4_d_out = self.layer4_d(self.layer3_d_out)

        self.middle_d_in = self.layer4_d_out
        self.middle_d_out = self.middle_d(self.layer5_d_out)
        self.middle_u_out = self.middle_u(self.middle_d_out)
        # Input to decoder layer 4, before concatenating skip connection
        self.in_layer4_u = self.middle_u_out

        in_4 = torch.cat([self.layer4_u, self.layer4_d_out], 1)
        self.layer4_u_out = self.layer4_u(in_4)
        in_3 = torch.cat([self.layer4_u_out, self.layer3_d_out], 1)
        self.layer3_u_out = self.layer3_u(in_3)
        in_2 = torch.cat([self.layer3_u_out, self.layer2_d_out], 1)
        self.layer2_u_out = self.layer2_u(in_2)
        in_1 = torch.cat([self.layer2_u_out, self.layer1_d_out], 1)
        self.layer1_u_out = self.layer1_u(in_1)
        in_0 = torch.cat([self.layer1_u_out, self.layer0_d_out], 1)
        self.layer0_u_out = self.layer0_u(in_0)
        return self.layer0_u_out


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


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # if len(gpu_ids) > 0:
    #    assert(torch.cuda.is_available())
    #    net.to(gpu_ids[0])
    #    net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    net = net.cuda()
    return net

def get_generator(input_nc, output_nc, ngf, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    netG = UnetGenerator(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(netG, init_type, init_gain, gpu_ids)


def get_descriminator(input_nc, ndf, use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    return init_net(netD, init_type, init_gain, gpu_ids)
