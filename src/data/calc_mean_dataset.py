from data.virtual_puzzle_dataset import VirtualPuzzleDataset
from data.virtual_image import VirtualImage
from globals import ORIENTATION_MAGIC, HORIZONTAL, NAME_MAGIC


class CalcMeanDataset(VirtualPuzzleDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = VirtualPuzzleDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(no_normalize=True)
        return parser

    def load_base_images(self):
        num_examples_accumulated = 0
        VirtualImage.initialize(self.opt)
        for path in [p for p in self.paths if ORIENTATION_MAGIC + HORIZONTAL in p and
                     NAME_MAGIC + str(self.opt.puzzle_name) in p]:
            current_image = VirtualImage(path, num_examples_accumulated)
            num_examples_accumulated = current_image.num_examples_accumulated
            self.images.append(current_image)

    def __getitem__(self, index):
        return {'image': self.images[index].horizontal, 'path': self.images[index].name_horizontal}

    def __len__(self):
        return len(self.images)

    def name(self):
        return 'CalcMeanDataset'







