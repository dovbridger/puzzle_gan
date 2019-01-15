import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from utils.visualizer import save_images, Visualizer
from utils import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        true_probability = model.get_probabilities()['True']
        false_probability = model.get_probabilities()['False']
        print("T:" + str(true_probability) + " , F:" + str(false_probability))
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % opt.save_images_frequency == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
