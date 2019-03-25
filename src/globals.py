from os import path

METADATA_FILE_NAME = 'metadata.txt'
METADATA_FOLDER_NAME = 'metadata'
METADATA_DELIMITER = ';'
WINDOWS_ROOT_DIR = r'C:\SHARE\checkouts\puzzle_gan_data'
DATASET_NAME = 'virtual_puzzle_parts'
MODEL_ROOT_DIR_NAME = "CNN_small_input"
ROOT_OF_MODEL_DATA = path.join(WINDOWS_ROOT_DIR, 'artifacts', MODEL_ROOT_DIR_NAME)
TEST_DATA_PATH = path.join(WINDOWS_ROOT_DIR, 'datasets', DATASET_NAME, "test")
FIGURES_FOLDER = path.join(ROOT_OF_MODEL_DATA, "figures")
NAME_MAGIC = 'name='
BURN_EXTENT_MAGIC = 'burn_extent='
PART_SIZE_MAGIC = 'part_size='
ORIENTATION_MAGIC = 'orientation='
HORIZONTAL = 'h'
VERTICAL = 'v'
DELIMITER_MAGIC = '-'
PART_SIZE = '64'
BURN_EXTENT = '4'
INPUT_IMAGE_TYPE = '.jpg'
NUM_DECIMAL_DIGITS = 5



