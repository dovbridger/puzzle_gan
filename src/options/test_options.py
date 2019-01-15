from .base_options import BaseOptions
from os import path


class TestOptions(BaseOptions):
    '''
    Append additional training options to the BaseOptions
    This class will be instantiated in 'test.py'
    '''
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default=path.join(self.saved_data_root, 'results'), help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, validation, test')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        parser.add_argument('--save_images_frequency', type=int, default=1, help='Frequency of saving test images')

        parser.set_defaults(model='test')
        # Deactivate visdom server
        parser.set_defaults(display_id=-1)
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        parser.set_defaults(only_crop=0)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(serial_batches=True)
        
        self.isTrain = False
        return parser
