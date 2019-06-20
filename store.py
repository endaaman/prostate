import argparse
import torch
import matplotlib.pyplot as plt

KEY_WEIGHTS = 'weights'
KEY_OPTIMS = 'optim_state'
KEY_LOSSES = 'losses'
KEY_DICES = 'dices'
KEY_IOUS = 'ious'

class Store():
    def __init__(self):
        self.weights = None
        self.optim_state = None
        self.losses = []
        self.dices = []
        self.ious = []

    def load(self, path):
        data = torch.load(path)
        self.weights = data.get(KEY_WEIGHTS)
        self.optim_state = data.get(KEY_OPTIMS)
        self.losses = data.get(KEY_LOSSES) or []
        self.dices = data.get(KEY_DICES) or []
        self.ious = data.get(KEY_IOUS) or []

    def append_params(self, weights, optim_state=None, loss=None, dice=None, iou=None):
        self.weights = weights
        if optim_state:
            self.optim_state = optim_state
        if loss:
            self.losses.append(loss)
        if dice:
            self.dices.append(dice)
        if iou:
            self.ious.append(iou)

    def save(self, path):
        torch.save({
            KEY_WEIGHTS: self.weights,
            KEY_OPTIMS: self.optim_state,
            KEY_LOSSES: self.losses,
            KEY_DICES: self.dices,
            KEY_IOUS: self.ious,
            }, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    PATH = args.path

    store = Store()
    store.load(PATH)

    def plot_line(plt, values, label, offset=10):
        plt.plot(values, label=label)
        for i, value in enumerate(values):
            text = "{:.2f}".format(value)
            plt.annotate(text, # this is the text
                         (i, value), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0, offset), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
    plt.figure(figsize=(max(len(store.losses)//1.5, 10), 10))
    plot_line(plt, store.losses, 'loss')
    plot_line(plt, store.dices, 'dice index', offset=-10)
    plot_line(plt, store.ious, 'IoU')
    plt.grid(True)
    plt.legend()
    plt.show()
