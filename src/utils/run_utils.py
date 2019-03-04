from puzzle.puzzle_utils import get_info_from_file_name
from globals import ORIENTATION_MAGIC, VERTICAL

def mixed_label_predictions(model):
    paths = model.get_image_paths()
    labels = []
    for path in paths:
        if 'True' in path:
            labels.append(1)
        elif 'False' in path:
             labels.append(0)
        else:
            print("Warning - Cannot determine label from path")
            labels.append(-1)
        predictions = model.get_prediction()
        print(predictions)
        return get_confusion_matrix_from_labled_predictions(predictions, labels)


def virtual_model_predictions(model):
    predictions, labels = model.get_prediction()
    return get_confusion_matrix_from_labled_predictions(predictions, labels)


def get_confusion_matrix_from_labled_predictions(predictions, labels):
    mistakes = [abs(prediction - label) > 0.5 for prediction, label in zip(predictions, labels)]
    classifications = ['negative' if prediction < 0.5 else 'positive' for prediction in predictions]
    correctnesses = ['false' if mistake else 'true' for mistake in mistakes]
    conclusions = [correctnes + '_' + classification for correctnes, classification in zip(correctnesses, classifications)]
    return predictions, conclusions


def adjust_image_width_for_vertical_image_in_webpage(image_path, opt):
    width = opt.display_winsize
    if get_info_from_file_name(image_path, ORIENTATION_MAGIC) == VERTICAL:
        width = width / 2
    return width
