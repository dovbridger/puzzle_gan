from data.puzzle_dataset import PuzzleDataset
import os.path
from data.image_folder import make_dataset


class PuzzleTestDataset(PuzzleDataset):
    '''
    An extended version of 'PuzzleDataset' That appends examples of false puzzle piece neighbors to qualitively compare
    the inpainting results to those of the true neighbors
    '''

    def modify_commandline_options(parser, is_train):
        assert not is_train, "PuzzleTestDataset should not be used in training mode"
        parser.add_argument('--num_false_examples', type=int, default=1,
                            help='What is the maximum number of false completion examples to test for each true example')

    def initialize(self, opt):
        super(PuzzleTestDataset, self).initialize(opt)
        # The folder containing images of false adjacent puzzle pieces according to 'opt.phase' (should bet 'test'))
        self.phase_folder_false = os.path.join(self.root, opt.phase, 'False')

        # Paths of the images of false adjacent puzzle pieces
        self.false_paths = sorted(make_dataset(self.phase_folder_false))

    def __getitem__(self, index):
        '''
        Retrieve an example containing g a pair of true neighbors, and possibly multiple examples of false neighbors of
        The same left piece in the true neighbors
        :param index: index of the requested example (examples are sorted according to path of the true neighbors)
        :return: A dict {'real_true' : The input image of true adjacent puzzle pieces'
                         'burnt_true': The burnt image of 'real_true' - missing some pixel columns between the two parts
                         'path': Path to the original image used to load 'real_true'
                         'real_false': A list containing between zero and 'opt.num_false_examples' images where the left
                                       'piece is the same as in 'real_true' and the right piece is a piece that is not
                                       the true neighbor in the puzzle
                         'burnt_false': A list of same size as 'real_false' containing the burn versions of the images
                         }
        '''
        # A path to example of true neighboring puzzle pieces
        path_true = self.true_paths[index]

        # A list paths to examples of the left puzzle piece from the true example,
        # adjacent to a puzzle piece on the right that is not the true neighbor
        paths_false = self.get_false_paths(path_true)
        real_image_true = self.get_real_image(path_true)
        burnt_image_true = self.burn_image(real_image_true)

        # List of images corresponding the to the paths 'paths_false'
        real_images_false = [self.get_real_image(path) for path in paths_false]

        # List containing the burnt version of each image in 'real_images_false'
        burnt_images_false = [self.burn_image(image) for image in real_images_false]
        return {'real_true': real_image_true, 'burnt_true': burnt_image_true, 'path': path_true,
                'burnt_false': burnt_images_false, 'real_false': real_images_false}

    def get_false_paths(self, true_path):
        '''
        Get a list paths to images of a pair of false neighbors
        :param true_path: A path to an image of a true neighbor pair of puzzle pieces on which to base the false pairs
                          (The false pairs will have a path that begins the same as the true pair, and the images will
                          share the same left piece as the true pair)
        :return:
        '''
        file_name_true_without_extension = os.path.basename(true_path).split('.')[0]
        false_paths = []
        for file in self.false_paths:
            if len(false_paths) >= self.opt.num_false_examples:
                return false_paths

            # The current file contains the same left puzzle piece as in the true example
            elif os.path.basename(file).startswith(file_name_true_without_extension):
                false_paths.append(file)
        return false_paths

    def name(self):
        return 'PuzzleTestDataset'
