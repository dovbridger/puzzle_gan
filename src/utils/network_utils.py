import torch
import os.path


def get_discriminator_input(opt, image):
    result = image[:, :, opt.burn_extent: -opt.burn_extent, opt.burn_extent: -opt.burn_extent]
    return crop_tensor_width(result, get_centered_window_indexes(opt.fineSize[1], opt.discriminator_window))


def crop_tensor_width(tensor4d, indexes):
    return tensor4d[:, :, :, indexes[0]:indexes[1]]


def get_centered_window_indexes(initial_size, required_window_size):
    center = int(initial_size / 2)
    window_start = center - int(required_window_size / 2)
    window_end = window_start + required_window_size
    return (window_start, window_end)


def get_generator_mask(input_size, generator_window, burn_extent):
    generator_mask = torch.zeros(input_size)
    centere_window_width_indexes = get_centered_window_indexes(input_size[1], generator_window)
    generator_mask[burn_extent: -burn_extent,
                   centere_window_width_indexes[0] : centere_window_width_indexes[1]] = 1
    return generator_mask

def get_network_file_name(which_epoch, name):
    return'%s_net_%s.pth' % (which_epoch, name)


def get_network_path(opt, network_name):
    return os.path.join(opt.checkpoints_dir,
                        opt.network_to_load,
                        get_network_file_name(opt.network_load_epoch, network_name))


def get_generator_path(opt):
    if opt.network_to_load is None:
        opt_file = os.path.join(opt.checkpoints_dir, opt.container_model, 'opt.txt')
        opt.network_to_load = get_option_from_opt_file(opt_file, 'network_to_load')
    return get_network_path(opt, 'G')


def get_option_from_opt_file(opt_file, option_name):
    with open(opt_file, 'r') as f:
        options = f.read()
    key = option_name + ': '
    key_start = options.find(key)
    if key_start == -1:
        return None
    option_start = key_start + len(key)
    option_end = option_start + options[option_start:].find(' ')
    return options[option_start:option_end]
