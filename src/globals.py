from os import path
is_CGM60 = path.exists(r"C:\is_cgm60.txt")
METADATA_FILE_NAME = 'metadata.txt'
METADATA_FOLDER_NAME = 'metadata'
METADATA_DELIMITER = ';'
CHECKOUTS_FOLDER = r"C:\SHARE\checkouts" if is_CGM60 else r"E:\checkouts" #r"C:\users\dov\checkouts"
WINDOWS_ROOT_DIR = path.join(CHECKOUTS_FOLDER, 'puzzle_gan_data')
DATASET_NAME = 'virtual_puzzle_parts' if is_CGM60 else 'MET'
MODEL_ROOT_DIR_NAME = "9_pieces"
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
PART_SIZE = '128'
BURN_EXTENT = '20'
LOAD_SIZE = (384, 384)
FINE_SIZE = (128, 256)
INPUT_IMAGE_TYPE = '.jpg'
NUM_DECIMAL_DIGITS = 5
DATASET_MEAN = [0.4509, 0.4372, 0.4059]
DATASET_STD = [0.2802, 0.2671, 0.2908]
SAVE_ALL_FIGURES = False




