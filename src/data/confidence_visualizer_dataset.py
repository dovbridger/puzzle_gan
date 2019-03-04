import os.path
import torch
from data.virtual_puzzle_dataset import VirtualPuzzleDataset
from puzzle.java_utils import PROBABILITY_MATRIX_FOLDER, resolve_orientation, DIRECTION_NAMES,\
    load_diff_matrix_cnn_from_probability
from puzzle.puzzle_utils import get_full_puzzle_name_from_characteristics, set_orientation_in_name
from models.calc_diff_model import CALC_DIFF_MODEL_NAME
from puzzle.scoring_statistics import get_top_false_confidence_score_debug_info
from globals import HORIZONTAL, VERTICAL, NUM_DECIMAL_DIGITS


class ConfidenceVisualizerDataset(VirtualPuzzleDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = VirtualPuzzleDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--probability_matrix_folder', type=str, default=PROBABILITY_MATRIX_FOLDER,
                            help="Folder where the diff matricies on which the visualization will be based on")
        parser.add_argument('--container_model', type=str,
                            help='Name of the model that created the diff matricies. Insures that the correct matrices'
                                 'and generator are used')
        parser.add_argument('--num_results', type=int, default=36, help='Number of result visualizations for each puzzle')

        return parser

    def initialize(self, opt):
        super(ConfidenceVisualizerDataset, self).initialize(opt)
        self.visualization_keys = self._get_visualization_keys()

    def _get_visualization_keys(self):
        for file in self._get_diff_matrix_files():
            puzzle_name, _ = os.path.splitext(file.split('-')[-1])
            probability_matrix_file = os.path.join(self.opt.probability_matrix_folder, file)
            diff_matrix = load_diff_matrix_cnn_from_probability(puzzle_name,
                                                                model_name=None,
                                                                file_name=probability_matrix_file,
                                                                use_log=self.opt.use_log_diff)

            image_name = get_full_puzzle_name_from_characteristics(puzzle_name, part_size=self.opt.part_size)
            metadata = self.get_image_metadata(image_name)
            top_false_confidence_score_info = \
                get_top_false_confidence_score_debug_info(diff_matrix,
                                                          num_x_parts=metadata.num_x_parts,
                                                          num_y_parts=metadata.num_y_parts,
                                                          num_results=self.opt.num_results)

            top_false_confidence_score_info = [(score, (image_name, direction, part1, part2s)) for
            score, (direction, part1, part2s) in top_false_confidence_score_info]
        return sorted(top_false_confidence_score_info, reverse=True)

    def _get_diff_matrix_files(self):
        wanted_files = [file for file in os.listdir(self.opt.probability_matrix_folder) if
                        CALC_DIFF_MODEL_NAME + "_" + self.opt.container_model + '-' in file]
        if self.opt.puzzle_name != '':
            wanted_files = [file for file in wanted_files if file.endswith('-' + self.opt.puzzle_name + '.json')]
        assert len(wanted_files) > 0, "No appropriate diff matrix file was found"
        return wanted_files

    def __getitem__(self, index):
        score, (image_name, direction, part1, part2s) = self.visualization_keys[index]
        burnt = torch.tensor([])
        text = []
        num_visuals = len(part2s)
        for i in range(num_visuals):
            image_name_prepared, part1_prepared, part2_prepared = self.prepare_pair_for_display(image_name,
                                                                                                direction,
                                                                                                part1,
                                                                                                part2s[i][0])
            current_example = self.get_pair_example_by_name(image_name_prepared, part1_prepared, part2_prepared)
            burnt = torch.cat((burnt, current_example['burnt'].unsqueeze(0)), 0)
            text.append((part1, part2s[i][0], round(part2s[i][1], NUM_DECIMAL_DIGITS)))
        name = "{0}-{1}, confidence={2}.ext".format(image_name_prepared,
                                                    DIRECTION_NAMES[direction],
                                                    round(score, NUM_DECIMAL_DIGITS))
        return {'score': score, 'burnt': burnt, 'text': text, 'size': num_visuals,
                'name': name}

    def __len__(self):
        return len(self.visualization_keys)

    def name(self):
        return 'ConfidenceVisualizerDataset'

    def prepare_pair_for_display(self, image_name, direction, part1, part2):
        metadata = self.get_image_metadata(image_name)
        part1, part2 = resolve_orientation(direction, part1, part2,
                                           num_x_parts=metadata.num_x_parts,
                                           num_y_parts=metadata.num_y_parts)
        orientation = VERTICAL if direction in [0, 1] else HORIZONTAL
        image_name = set_orientation_in_name(image_name, orientation)
        return image_name, part1, part2





