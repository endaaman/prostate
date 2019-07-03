import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from metrics import Metrics


KEY_WEIGHTS = 'weights'
KEY_OPTIMS = 'optims'
KEY_METRICS = 'metrics'

class Store():
    def __init__(self):
        self.weights = None
        self.optims = None
        self.metrics = None

    def load(self, path, map_location=None):
        data = torch.load(path, map_location=map_location)
        self.weights = data.get(KEY_WEIGHTS)
        self.optims = data.get(KEY_OPTIMS)
        self.metrics = data.get(KEY_METRICS)

    def set_states(self, weights, optims, metrics):
        assert weights
        assert optims
        assert metrics
        self.weights = weights
        self.optims = optims
        self.metrics = metrics

    def save(self, path):
        torch.save({
            KEY_WEIGHTS: self.weights,
            KEY_OPTIMS: self.optims,
            KEY_METRICS: self.metrics,
            }, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    PATH = args.path

    store = Store()
    store.load(PATH, map_location='cpu')

    metrics = Metrics()
    metrics.load_state_dict(store.metrics)

    def plot_line(plt, values, label, offset=None):
        plt.plot(values, label=label)
        if not offset:
            return
        for i, value in enumerate(values):
            text = "{:.3f}".format(value)
            plt.annotate(text, # this is the text
                         (i, value), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0, offset), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center

    epoch = len(metrics.get('losses'))
    plt.figure(figsize=(max(epoch//1.5, 10), 10))

    name = os.path.splitext(os.path.basename(PATH))[0]
    plt.title(name)

    plot_line(plt, metrics.get('losses'), 'loss')
    plot_line(plt, metrics.get('jacs'), 'IoU')
    plot_line(plt, metrics.get('pjacs'), 'pIoU')
    plot_line(plt, metrics.get('pdices'), 'acc')

    plot_line(plt, metrics.get('gsensis'), 'gland sensi')
    plot_line(plt, metrics.get('gspecs'), 'gland specs')
    plot_line(plt, metrics.get('tsensis'), 'tumor sensi')
    plot_line(plt, metrics.get('tspecs'), 'tumor specs')

    plt.xticks(list(range(0, epoch)))
    plt.yticks(np.arange(0, 11) / 10)
    plt.grid(True)
    plt.legend()
    plt.show()
