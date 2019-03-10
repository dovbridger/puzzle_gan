from data.virtual_puzzle_dataset import VirtualPuzzleDataset
from data.base_dataset import get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms


class CalcMeanDataset(VirtualPuzzleDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return VirtualPuzzleDataset.modify_commandline_options(parser, is_train)

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.images = []
        self.images_index_dict = {}
        self.phase_folder = self.root

        # Paths of the full puzzle images
        self.paths = sorted(make_dataset(self.phase_folder))
        self.transform = get_transform(opt)
        self.load_base_images()

    def __getitem__(self, index):
        return {'image': self.images[index].horizontal, 'path': self.images[index].name_horizontal}

    def __len__(self):
        return len(self.images)

    def get_real_image(self, path):
        original_img = Image.open(path).convert('RGB')
        # Perform the transformations
        real_image = self.transform(original_img)

        return real_image

    def name(self):
        return 'CalcMeanDataset'







