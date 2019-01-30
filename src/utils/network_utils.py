import torch
import os.path


def get_discriminator_input(opt, burnt_image, image):
    if opt.provide_burnt:
        result = torch.cat((burnt_image, image), 1)
    else:
        result = image

    return crop_tensor_width(result, get_centered_window_indexes(opt.fineSize[1], opt.discriminator_window))


def crop_tensor_width(tensor4d, indexes):
    return tensor4d[:, :, :, indexes[0]:indexes[1]]


def get_centered_window_indexes(initial_size, required_window_size):
    center = int(initial_size / 2)
    window_start = center - int(required_window_size / 2)
    window_end = window_start + required_window_size
    return (window_start, window_end)


def get_network_file_name(which_epoch, name):
    return'%s_net_%s.pth' % (which_epoch, name)
