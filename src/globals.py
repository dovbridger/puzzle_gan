from os import path

METADATA_FILE_NAME = 'metadata.txt'
WINDOWS_ROOT_DIR = r"C:\SHARE"
MODEL_ROOT_DIR_NAME = "CNN_small_input"
ROOT_OF_MODEL_DATA = path.join(WINDOWS_ROOT_DIR, MODEL_ROOT_DIR_NAME)
TRAINING_DATA_PATH = path.join(ROOT_OF_MODEL_DATA, "train")
TEMP_DATA_PATH = path.join(ROOT_OF_MODEL_DATA, "temp")
VALIDATION_DATA_PATH = path.join(ROOT_OF_MODEL_DATA, "validation")
TEST_DATA_PATH = path.join(ROOT_OF_MODEL_DATA, "test")
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



