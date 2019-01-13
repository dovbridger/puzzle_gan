from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')
        parser.set_defaults(dataset_name='puzzle_test')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['burnt_true', 'real_true', 'fake_true']
        for i in range(opt.num_false_examples):
            self.visual_names.append('burnt_false_' + str(i))
            self.visual_names.append('real_false_' + str(i))
            self.visual_names.append('fake_false_' + str(i))
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]

        self.netG = networks.get_generator(opt.input_nc, opt.output_nc, opt.ngf, opt.init_type, opt.init_gain,
                                           self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        self.real_true = input['real_true'].to(self.device)
        self.burnt_true = input['burnt_true'].to(self.device)
        real_false_images = input['real_false']
        burnt_false_images = input['burnt_false']
        for i in range(self.opt.num_false_examples):
            if i < len(burnt_false_images):
                setattr(self, 'burnt_false_' + str(i), burnt_false_images[i].to(self.device))
                setattr(self, 'real_false_' + str(i), real_false_images[i].to(self.device))
            else:
                setattr(self, 'burnt_false_' + str(i), None)
                setattr(self, 'real_false_' + str(i), None)

        self.image_paths = input['path']

    def forward(self):
        self.fake_true = self.netG(self.burnt_true)
        for i in range(self.opt.num_false_examples):
            burnt_false = getattr(self, 'burnt_false_' + str(i))
            if burnt_false is not None:
                setattr(self, 'fake_false_' + str(i), self.netG(burnt_false))
            else:
                setattr(self, 'fake_false_' + str(i), None)