"""Plot"""

import numpy as np
import matplotlib.pyplot as plt

from farms_core.analysis.plot import (
    plt_legend_side,
    colorgraph,
    plot2d,
    grid,
)


def plot_trajectories(plots, exp_data, legend=True, **kwargs):
    """Plot trajectories"""
    plot_name = kwargs.pop('plot_name')
    condition = kwargs.pop('condition', lambda _: True)
    max_labels_per_row = kwargs.pop('max_labels_per_row', 20)
    plots[plot_name] = plt.figure(plot_name)
    n_labels = 0
    for label_i, (label, data) in enumerate(exp_data.items()):
        # prefix = f'{label} - ' if len(exp_data) > 1 else ''
        for exp in data:
            if condition(exp['drive']):
                plt.plot(
                    exp['positions'][:, 0],
                    exp['positions'][:, 1],
                    # label=f'{prefix}Drive={exp["drive"]}',
                    label=label,
                    zorder=10-label_i,
                )
                n_labels += 1
                grid()
    if legend and n_labels > 1:
        plt_legend_side(n_labels, max_labels_per_row)
    plt.xlabel('Position X [m]')
    plt.ylabel('Position Y [m]')
    plt.gca().set_aspect('equal')


def plot_element(plots, exp_data, **kwargs):
    """Plot element"""
    plot_name = kwargs.pop('plot_name')
    xdata = kwargs.pop('xdata')
    ydata = kwargs.pop('ydata')
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    ylim = kwargs.pop('ylim', None)
    zorder_i = kwargs.pop('zorder_i', 0)
    label_template = kwargs.pop('label_template', '{label}')
    show_legend = kwargs.pop('show_legend', False)
    condition = kwargs.pop('condition', lambda _: True)
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = '-'
    if 'marker' not in kwargs:
        kwargs['marker'] = '.'
    plots[plot_name] = plt.figure(plot_name)
    for label_i, (label, data) in enumerate(exp_data.items()):
        plt.plot(
            [exp[xdata] for exp in data if condition(exp['drive'])],
            [exp[ydata] for exp in data if condition(exp['drive'])],
            label=label_template.format(label=label),
            zorder=10*(10-label_i)-zorder_i,
            **kwargs,
        )
        grid()
    if show_legend or len(exp_data) > 1:
        plt_legend_side(n_labels=len(exp_data), max_labels_per_row=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)


def plot_multi_exponent(plots, exp_data, **kwargs):
    """Plot multi-exponent"""
    plot_name_template = kwargs.pop('plot_name')
    xdata = kwargs.pop('xdata')
    ydata = kwargs.pop('ydata')
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    equation = kwargs.pop('equation')
    show_legend = kwargs.pop('show_legend', False)
    condition = kwargs.pop('condition', lambda _: True)
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = '-'
    if 'marker' not in kwargs:
        kwargs['marker'] = '.'
    for exponent in range(1, 5):
        plot_name = plot_name_template.format(exponent)
        plots[plot_name] = plt.figure(plot_name)
        for label_i, (label, data) in enumerate(exp_data.items()):
            plt.plot(
                [
                    exp[xdata]
                    for exp in data
                    if condition(exp['drive'])
                ],
                [
                    exp[ydata.format(exponent)]
                    for exp in data
                    if condition(exp['drive'])
                ],
                label=label,
                zorder=10-label_i,
                **kwargs,
            )
            grid()
        if show_legend or len(exp_data) > 1:
            plt_legend_side(n_labels=4*len(exp_data), max_labels_per_row=20)
        plt.xlabel(xlabel)
        exp = f'^{exponent}' if exponent > 1 else ''
        plt.ylabel(ylabel.format(equation=equation.format(exp=exp)))


def plot_element_2d(plots, exp_data, **kwargs):
    """Plot element"""
    plot_name = kwargs.pop('plot_name')
    xdata = kwargs.pop('xdata')
    ydata = kwargs.pop('ydata')
    zdata = kwargs.pop('zdata')
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    zlabel = kwargs.pop('zlabel')
    condition = kwargs.pop('condition', lambda _: True)
    plots[plot_name] = plt.figure(plot_name)
    plot2d(
        results=np.array([
            [exp[xdata], exp[ydata], exp[zdata]]
            for data in exp_data.values()
            for exp in data
            if condition(exp['drive'])
        ]),
        labels=[xlabel, ylabel, zlabel],
        **kwargs,
    )


def plot_multi_exponent_2d(plots, exp_data, **kwargs):
    """Plot multi-exponent"""
    plot_name_template = kwargs.pop('plot_name')
    xdata = kwargs.pop('xdata')
    ydata = kwargs.pop('ydata')
    zdata = kwargs.pop('zdata')
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    zlabel = kwargs.pop('zlabel')
    equation = kwargs.pop('equation')
    condition = kwargs.pop('condition', lambda _: True)
    for exponent in range(1, 5):
        plot_name = plot_name_template.format(exponent)
        plots[plot_name] = plt.figure(plot_name)
        exp = f'^{exponent}' if exponent > 1 else ''
        plot2d(
            results=np.array([
                [exp[xdata], exp[ydata], exp[zdata.format(exponent)]]
                for data in exp_data.values()
                for exp in data
                if condition(exp['drive'])
            ]),
            labels=[
                xlabel,
                ylabel,
                zlabel.format(equation=equation.format(exp=exp)),
            ],
            **kwargs,
        )


def plot_element_colorgraph(plots, exp_data, size_auto=False, **kwargs):
    """Plot element"""
    plot_name = kwargs.pop('plot_name')
    xdata = kwargs.pop('xdata')
    ydata = kwargs.pop('ydata')
    condition = kwargs.pop('condition', lambda _: True)
    fig = plots[plot_name] = plt.figure(plot_name)
    data = [
        exp[xdata]
        for data in exp_data.values()
        for exp in data
        if condition(exp['drive'])
    ]
    labels = [
        exp[ydata]
        for data in exp_data.values()
        for exp in data
        if condition(exp['drive'])
    ]
    assert len(data) == len(labels), f'{len(data)=} != {len(labels)=}'
    if size_auto:
        height = 0.7*len(labels)
        size = fig.get_size_inches()
        aspect = kwargs.get('aspect', None)
        width = kwargs.get('width', height/aspect if aspect else size[0])
        fig.set_size_inches(width, height)
    colorgraph(
        data=data,
        labels=labels,
        **kwargs,
    )
    grid()
