import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from utils.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    # key - action to be performed, value - the value the step counter needs to reach before performing the action
    next_action = {'display': opt.display_freq, 'print': 0, 'update_html': opt.update_html_freq,
                   'save_latest': opt.save_latest_freq, 'save_epoch': opt.save_epoch_freq}
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            if total_steps >= next_action['display']:
                save_result = False
                if total_steps >= next_action['update_html']:
                    save_result = True
                    next_action['update_html'] = next_action['update_html'] + opt.update_html_freq
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                next_action['display'] = next_action['display'] + opt.display_freq

            if total_steps >= next_action['print']:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, iter_start_time - iter_data_time)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
                next_action['print'] = next_action['print'] + opt.print_freq

            if total_steps >= next_action['save_latest']:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')
                next_action['save_latest'] = next_action['save_latest'] + opt.save_latest_freq

            iter_data_time = time.time()
        if epoch >= next_action['save_epoch']:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
            next_action['save_epoch'] = next_action['save_epoch'] + opt.save_epoch_freq

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
