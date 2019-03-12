from puzzle.puzzle_utils import get_info_from_file_name
from globals import ORIENTATION_MAGIC, VERTICAL, NUM_DECIMAL_DIGITS

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


def update_loss_count(model, loss_stats):
    losses = model.get_current_losses()
    for loss_name, loss_value in losses.items():
        loss_stats[loss_name] += loss_value
    return losses


def calc_loss_stats(losses):
    image_text = {key: round(value, NUM_DECIMAL_DIGITS) for key, value in losses.items()}
    min_loss = min(image_text.values())
    max_loss = max(image_text.values())
    for key, value in image_text.items():
        if value == min_loss:
            image_text[key] = (value, 'green')
        elif value == max_loss:
            image_text[key] = (value, 'red')
        else:
            image_text[key] = (value, 'black')
    return image_text
