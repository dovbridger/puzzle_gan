import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from utils.visualizer import save_images, Visualizer
from utils.plot_utils import plot_discriminator_results, print_loss_stats, print_confusion_matrix
from utils import html
import json
from utils.run_utils import mixed_label_predictions, virtual_model_predictions, calc_loss_stats, update_loss_count,\
    adjust_image_width_for_vertical_image_in_webpage
from globals import NUM_DECIMAL_DIGITS

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 0
    opt.batchSize = 1  # test code only supports batchSize = 1
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s' % (opt.phase, opt.which_epoch, opt.experiment_name))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    discriminator_results_file = os.path.join(web_dir, 'discriminator_results.json')
    loss_stats_file = os.path.join(web_dir, 'loss_stats.txt')
    cm_file = os.path.join(web_dir, 'cm.txt')
    if opt.calc_loss_stats:
        loss_stats = {loss_name: 0 for loss_name in model.loss_names}
        loss_items_count = 0
        assert len(loss_stats) > 0, "'calc_loss_stats' is True but there are no losses"
    probability_results = {'True': [], 'False': []}
    cm = {key: 0 for key in ['false_positive', 'false_negative', 'true_positive', 'true_negative']}
    mistake_count = 0
    # Run the test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        mistake = False
        if opt.discriminator_test:
            if opt.dataset_name == 'puzzle_with_false':
                probabilities = model.get_probabilities()
                for key in probabilities:
                    probability_results[key].append(probabilities[key].tolist())
                    true_probablitiy = round(probabilities['True'].mean().item(), NUM_DECIMAL_DIGITS)
                    false_probability = round(probabilities['False'].mean().item(), NUM_DECIMAL_DIGITS)
                    image_text = {'fake_true': (true_probablitiy, 'green' if true_probablitiy >= 0.5 else 'red'),
                                  'fake_false_0': (false_probability, 'green' if false_probability < 0.5 else 'red')}
                    mistake = true_probablitiy < 0.5 or false_probability >= 0.5
            else:
                if opt.dataset_name == 'virtual_puzzle':
                    predictions, conclusions = virtual_model_predictions(model)
                else:
                    predictions, conclusions = mixed_label_predictions(model)
                mistake = 'false' in conclusions[0]
                for conclusion in conclusions:
                    cm[conclusion] += 1
                image_text = {'fake': (round(predictions[0].item(), NUM_DECIMAL_DIGITS), 'red' if mistake else 'green')}
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        losses = None
        if opt.calc_loss_stats:
            losses = update_loss_count(model, loss_stats)
            loss_items_count += 1
        if i % opt.save_images_frequency == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
            if opt.calc_loss_stats:
                image_text = calc_loss_stats(losses)
            mistake_count += int(mistake)
            if mistake or not opt.save_mistakes_only:
                width = adjust_image_width_for_vertical_image_in_webpage(img_path[0], opt)
                save_images(webpage, visuals, img_path, additional_texts=image_text, aspect_ratio=opt.aspect_ratio, width=width)
    if opt.discriminator_test:
        if opt.dataset_name == 'puzzle_with_false':
            with open(discriminator_results_file, 'w') as f:
                json.dump(probability_results, f)
            plot_discriminator_results(discriminator_results_file)
        else:
            print_confusion_matrix(cm, cm_file)
    if opt.calc_loss_stats:
        loss_stats = {loss_name: round(loss_value / loss_items_count, NUM_DECIMAL_DIGITS) for loss_name, loss_value in loss_stats.items()}
        print_loss_stats(loss_stats, loss_stats_file)
    print("Accuracy: {0}".format(float(i + 1 - mistake_count) / (i + 1)))
    webpage.save()
