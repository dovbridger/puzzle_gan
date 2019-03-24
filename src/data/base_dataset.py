import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import json


class BaseDataset(data.Dataset):
    '''
    Base class for future dataset classes
    '''
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        '''
        Implement this method to add / modify the command line options when this dataset class is used
        :param parser:
        :param is_train: boolean, is the script run in training mode
        :return: parser containing the modified commandline options
        '''
        return parser

    def initialize(self, opt):
        pass

    def __len__(self):
        return 0


def get_transform(opt):
    '''
    Builds and returns the transformations that need to be done on the input images according to the options specified
    in 'opt'
    :param opt: The command line options
    :return: A sequence of transforms to be performed on the input before feeding it to the network
    '''
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        if opt.only_crop == 0:
            transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        else:
            transform_list.append(transforms.CenterCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize[1])))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize[1])))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'none':
        pass

    elif opt.resize_or_crop == 'crop_to_part_size':
        transform_list.append(transforms.Lambda(
            lambda image: __inffer_crop_transform_from_part_size(image, opt.part_size)))
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize(get_normalization_parameters(opt.dataset_mean),
                                            get_normalization_parameters(opt.dataset_std))]
    return transforms.Compose(transform_list)


# just modify the width and height to be multiple of 4
def __adjust(img):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __inffer_crop_transform_from_part_size(image, part_size):
    width, height = image.size
    new_width = part_size * int(width / part_size)
    new_height = part_size * int(height / part_size)
    if (new_height, new_width) == (height, width):
        return image
    else:
        return transforms.functional.crop(image, 0, 0, new_height, new_width)


def get_normalization_parameters(param_str):
    try:
        param = json.loads(param_str)
    except Exception:
        print("invalid json string for dataset statistics: %s" % param_str)
        raise
    assert len(param) == 3, "dataset statistics param {0} is invalid, must have length 3".format(param_str)
    for channel in param:
        assert channel >= 0, "dataset statistics param {0} must contain only positive values".format(param_str)
    return param


def __scale_width(img, target_width):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


