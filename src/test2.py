import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from utils.visualizer import save_images, Visualizer
from utils import html
from utils.run_utils import adjust_image_width_for_vertical_image_in_webpage

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.batchSize = 1  # test code only supports batchSize = 1
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s' % (opt.phase, opt.which_epoch, opt.experiment_name))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # Run the test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        model.rotate_if_vertical()
        visuals = model.get_current_visuals()
        color = ['black', 'green', 'green'] if data['text'][-1] == data['text'][-2] else ['black', 'red', 'green']
        image_text = {model.visual_names[i]: ("{0}_{1}, diff={2}".format(data['text'][i][0].item(),
                                                                   data['text'][i][1].item(),
                                                                   data['text'][i][2].item()),
                                              color[i]) for i in range(data['size'])}
        img_path = model.get_image_paths()
        width = adjust_image_width_for_vertical_image_in_webpage(img_path[0], opt)
        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, additional_texts=image_text, aspect_ratio=opt.aspect_ratio, width=width)

    webpage.save()
