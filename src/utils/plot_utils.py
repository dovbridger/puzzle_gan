import numpy as np
import matplotlib.pyplot as plt
import os

def plot_images(ims, figsize=(12, 6), rows=1, interp=False, titles=None, colors=None, output_file_name=None):
    if colors is None:
        colors = ['black' for i in range(len(ims))]
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims.astype(np.uint8))
        #BUG? ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16, color=colors[i])
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    if output_file_name is not None:
        plt.savefig(output_file_name)
    plt.show()


def _get_axes_from_data(data_array):
    fig, axes = plt.subplots(1, len(data_array), sharey=True, tight_layout=True)
    if len(data_array) == 1:
        axes = [axes]
    return axes


def listify_input(f):
    def inner(x, *args, **kwargs):
        if isinstance(x, list):
            return f(x, *args, **kwargs)
        else:
            return f([x], *args, **kwargs)
    return inner


@listify_input
def plot_histograms(data, num_bins=20, titles=None, colors=None, labels=None, output_file_name=None):
        def hist_on_axes(axes, data, color=None, label=None, **kwargs):
            axes.hist(data, bins=num_bins, color=color, histtype='step', label=label)

        general_plot(data, hist_on_axes, configure_axes, titles=titles,
                     labels=labels, colors=colors, output_file_name=output_file_name)


@listify_input
def plot_bars(data, titles=None, colors=None, labels=None, tick_labels=None, bar_width=0.4,
              data_spacing=1.25, output_file_name=None):
    def bars_on_axes(axes, specific_label_data, index=0, num_data_items=1,
                     x_positions=None, color=None, label=None, **kwargs):
        if x_positions is None:
            x_positions = bar_width * (data_spacing * num_data_items * np.arange(len(specific_label_data)) + index)
        axes.bar(x_positions, height=specific_label_data, color=color, width=bar_width, label=label)

    general_plot(data, bars_on_axes, configure_axes, titles=titles,
                 labels=labels, colors=colors, tick_labels=tick_labels,
                 bar_width=bar_width, output_file_name=output_file_name)


def general_plot(data, plot_function, axes_configurator, output_file_name=None, **kwargs):
    '''

    :param data:
    :param plot_function: f(axes, data, ...)
    :param axes_configurator: f(index, list(axes), ...)
    :param kwargs:
    :return:
    '''
    axes = _get_axes_from_data(data)
    for i in range(len(data)):
        current_data = data[i]
        if isinstance(current_data, tuple):
            num_data_items = len(current_data)
            for index in range(num_data_items):
                indexed_kwargs = {key[:-1]: _get_property(kwargs, key, index) for key in kwargs if key.endswith('s')}
                additional_kwargs = {key: _get_property(kwargs, key, index) for key in kwargs if not key.endswith('s')}
                for key in additional_kwargs:
                    indexed_kwargs[key] = additional_kwargs[key]
                plot_function(axes[i], current_data[index], index=index, num_data_items=num_data_items, **indexed_kwargs)
        else:
            plot_function(axes[i], current_data, **kwargs)
        axes_configurator(i, axes, data_per_plot=len(current_data), **kwargs)
    if output_file_name is not None:
        if os.path.exists(output_file_name):
            file_name, extension = os.path.splitext(output_file_name)
            output_file_name = file_name + "_new." + extension
        plt.savefig(output_file_name)
    else:
        plt.show()


def configure_axes(index, axes, titles=None, tick_labels=None, bar_width=0.4, data_per_plot=1, **kwargs):
    if titles is not None:
        axes[index].set(title=titles[index])

    if tick_labels is not None:
        tick_width = data_per_plot * bar_width
        axes[index].set_xticks((np.arange(len(tick_labels)) + 0.5) * tick_width)
        axes[index].set_xticklabels(tick_labels)

    axes[index].legend()


def print_loss_stats(loss_stats, output_file_name):
    def print_dict(dict, fd):
        for key, value in dict.items():
            fd.write("{0} : {1}\n".format(str(key), str(value)))
        fd.write('\n')

    loss_stats_relative = {name: round(100 * min(loss_stats.values()) / value, 1) for name, value in loss_stats.items()}
    with open(output_file_name, 'w') as f:
        f.write('-------------- Average loss --------------------\n')
        print_dict(loss_stats, f)
        f.write('-------------- Relative loss --------------------\n')
        print_dict(loss_stats_relative, f)

def print_confusion_matrix(results, output_file_name):
    num_right = results['true_positive'] + results['true_negative']
    all_examples = num_right + results['false_positive'] + results['false_negative']
    accuracy = float(num_right) / float(all_examples)
    with open(output_file_name, 'w') as f:
        f.write('True Positives: {0}\n'.format(results['true_positive']))
        f.write('False Positives: {0}\n'.format(results['false_positive']))
        f.write('True Negatives: {0}\n'.format(results['true_negative']))
        f.write('False Negatives: {0}\n'.format(results['false_negative']))
        f.write('Accuracy: {0} out of {1} ({2})\n'.format(num_right, all_examples, accuracy))

def _get_property(dictionary, property_name, index=None):
    if property_name not in dictionary:
        return None
    value = dictionary[property_name]
    if not isinstance(value, list):
        return value
    if index is None:
        return value
    if index in range(len(value)):
        return value[index]
    return None


@listify_input
def plot_y(data, sort=False, titles=None, colors=None, labels=None, output_file_name=None):
    '''
    Plots xy plot where x is 0,1,2... and y is the data.
    :param data: data to be plotted, or a list of data to be plotted for comparison
    :param sort: Whether or not the data should be sorted
    :return:
    '''

    def plot_on_axes(axes, data, sort=False, color=None, label=None, **kwargs):
        if sort:
            data.sort()
        axes.plot(data, color=color, label=label)

    general_plot(data, plot_on_axes, configure_axes, sort=sort, titles=titles, colors=colors, labels=labels,
                 output_file_name=output_file_name)


def _parse_loss_file(file_name):
    def get_loss_names(loss_section):
        words = loss_section.split(' ')[:-1]
        return [words[i] for i in range(0, len(words), 2)]

    def get_loss_section(line):
        index = line.find(')') + 2
        return line[index:]
    with open(file_name, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    title = lines[0]
    loss_names = get_loss_names(get_loss_section(lines[1]))
    loss_sections = [get_loss_section(line) for line in lines[1:]]
    loss_values = {key: [] for key in loss_names}
    for section in loss_sections:
        section = section.split(' ')[:-1]
        for i in range(0, len(section), 2):
            loss_values[section[i]].append(float(section[i + 1]))
    values = tuple(np.array(loss_values[loss_name]) for loss_name in loss_names)
    return values, loss_names, title

def plot_loss_file(file_name):
    data_to_plot, loss_names, title = _parse_loss_file(file_name)
    plot_y(data_to_plot, titles=[title], labels=loss_names)

def _parse_discriminator_results(results_file):
    import json
    with open(results_file, 'r') as f:
        results = json.load(f)
    keys = list(results.keys())
    num_examples, _, _, height, width = np.array(results[keys[0]]).shape
    return [np.array(results[key]).reshape(num_examples, height, width) for key in keys], keys

def plot_discriminator_results(results_file, only_mean=True):
    results, labels = _parse_discriminator_results(results_file)
    num_examples, height, width = results[0].shape
    hist_folder = os.path.join(os.path.dirname(results_file), "histogram_results")
    if not os.path.exists(hist_folder):
        os.makedirs(hist_folder)
    if not only_mean:
        for i in range(height):
            for j in range(width):
                current_result = tuple(array[:, i, j] for array in results)
                plot_histograms(current_result, num_bins=10, titles=[str((i, j))], labels=labels,
                                output_file_name=os.path.join(hist_folder, str(i) + str(j) + ".jpg"))
    results_mean = tuple(array.reshape(num_examples, width * height).mean(axis=1) for array in results)
    plot_histograms(results_mean, num_bins=10, titles=['Mean'], labels=labels,
                    output_file_name=os.path.join(hist_folder, "mean.jpg"))

def main():
    file_name = r"C:\SHARE\checkouts\puzzle_gan\saved_data\results\batch1_post\test_1\discriminator_results.json"
    plot_discriminator_results(file_name)
if __name__ == '__main__':
    main()