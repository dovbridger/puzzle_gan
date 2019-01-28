import os.path
import torch
from data.base_dataset import PuzzleDatatset
from data.image_folder import make_dataset
from PIL import Image


class PuzzleWithOldInpaintingDataset(PuzzleDataset):
    '''
    Dov's implementation of a dataset to be used for the puzzle problem with access to old inpainting results
    '''
    def initialize(self, opt):
        super(PuzzleWithOldInpaintingDataset, self).initialize(opt)
        self.path_to_old_inpainting = os.path.join(self.root, 'old_inpainting', 'True')

    def __getitem__(self, index):
        result = super(PuzzleWithOldInpaintingDataset, self).__getitem__(index)
        old_inpainting_image_path = os.path.join(self.path_to_old_inpainting, os.path.basename(result['path']))
        result['old_inpainting'] = get_real_image(old_inpainting_image_path)
        return result

    def name(self):
        return 'PuzzleWithOldInpaintingDataset'
