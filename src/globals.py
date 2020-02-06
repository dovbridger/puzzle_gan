from os import path
CODE_ROOT_DIR = r"C:\SHARE\Puzzle GAN Code"
METADATA_FILE_NAME = 'metadata.txt'
METADATA_FOLDER_NAME = 'metadata'
METADATA_DELIMITER = ';'
DATA_ROOT_DIR = path.join(CODE_ROOT_DIR, 'puzzle_gan_data')
DATASET_NAME = 'virtual_puzzle_parts'
MODEL_ROOT_DIR_NAME = "Root Model"
ROOT_OF_MODEL_DATA = path.join(DATA_ROOT_DIR, 'artifacts', MODEL_ROOT_DIR_NAME)
TEST_DATA_PATH = path.join(DATA_ROOT_DIR, 'datasets', DATASET_NAME, "test")
FIGURES_FOLDER = path.join(ROOT_OF_MODEL_DATA, "figures")
NAME_MAGIC = 'name='
BURN_EXTENT_MAGIC = 'burn_extent='
PART_SIZE_MAGIC = 'part_size='
ORIENTATION_MAGIC = 'orientation='
HORIZONTAL = 'h'
VERTICAL = 'v'
DELIMITER_MAGIC = '-'
PART_SIZE = '64'
BURN_EXTENT = '2'
LOAD_SIZE = (64, 128)
FINE_SIZE = (64, 128)
INPUT_IMAGE_TYPE = '.jpg'
NUM_DECIMAL_DIGITS = 5
DATASET_MEAN = [0.5, 0.5, 0.5]
DATASET_STD = [0.5, 0.5, 0.5]
SAVE_ALL_FIGURES = False




