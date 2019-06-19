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
        self.optims = None
        self.losses = []
        self.dices = []
        self.ious = []

    def load(self, path):
        data = torch.load(path)
        self.weights = data[KEY_WEIGHTS]
        self.optim_state = data[KEY_OPTIMS]
        self.losses = data[KEY_LOSSES]
        self.dices = data[KEY_DICES]
        self.ious = data[KEY_IOUS]

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
    plt.plot(store.losses, label='loss')
    plt.plot(store.dices, label='dice index')
    plt.plot(store.ious, label='IoU')
    plt.grid(True)
    plt.legend()
    plt.show()
