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
def plot_histograms(data, num_bins=20):
    '''
    :param data: data to be plotted, or a list of data to be plotted for comparison
    :param num_bins: Hitogram bins
    :return:
    '''
    axes = _get_axes_from_data(data)
    for i in range(len(data)):
        axes[i].hist(data[i], bins=num_bins)
    plt.show()


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
            file_name, extension = output_file_name.split('.')
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



#def main():
    #plot_bars([([8, 1], [7, 2]), ([10, 9], [5, 4])], titles=['Horizontal', 'Vertical'], tick_labels=['True', 'False'], colors=['b','g'], labels=['original', 'model'])

