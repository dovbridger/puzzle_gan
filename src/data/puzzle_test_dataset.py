from data.puzzle_dataset import PuzzleDataset

class PuzzleTestDataset(PuzzleDataset):

    def __getitem__(self, index):
        path_true = self.true_paths[index]
        paths_false = self.get_false_paths(path_true)
        real_image_true = self.get_real_image(path_true)
        burnt_image_true = self.burn_image(real_image_true)
        real_images_false = [self.get_real_image(path) for path in paths_false]
        burnt_images_false = [self.burn_image(image) for image in real_images_false]
        return {'real_true': real_image_true, 'burnt_true': burnt_image_true, 'path': path_true,
                'burnt_false': burnt_images_false, 'real_false': real_images_false}

    def name(self):
        return 'PuzzleTestDataset'
