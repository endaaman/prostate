import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


KEY_WEIGHTS = 'weights'
KEY_OPTIMS = 'optims'
KEY_METRICS = 'metrics'

class Store():
    def __init__(self, name=None):
        self.name = name
        self.weights = None
        self.optims = None
        self.metrics = None

    def load(self, path, map_location=None):
        data = torch.load(path, map_location=map_location)
        self.weights = data.get(KEY_WEIGHTS)
        self.optims = data.get(KEY_OPTIMS)
        self.metrics = data.get(KEY_METRICS)

    def set_name(self, name):
        self.name = name

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

    def plot_line(plt, values, label, offset=10):
        plt.plot(values, label=label)
        for i, value in enumerate(values):
            text = "{:.3f}".format(value)
            plt.annotate(text, # this is the text
                         (i, value), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0, offset), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
    plt.figure(figsize=(max(len(store.losses)//1.5, 10), 10))
    plot_line(plt, store.losses, 'loss')
    plot_line(plt, store.dices, 'dice index', offset=-10)
    plot_line(plt, store.ious, 'IoU')

    if store.name:
        plt.title(store.name)
    plt.xticks(list(range(0, len(store.losses))))
    plt.yticks(np.arange(1,10) / 10)
    plt.grid(True)
    plt.legend()
    plt.show()
