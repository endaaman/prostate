import copy
import argparse
from collections import namedtuple, OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt

from formula import *
from utils import revert_onehot, similarity_index, pixel_similarity_index, inspection_accuracy


PLURAL_KEYS = ['losses', 'dices', 'jacs', 'pdices', 'pjacs', 'gsensis', 'gspecs', 'tsensis', 'tspecs',]
SINGULAR_KEYS = ['loss', 'dice', 'jac', 'pdice', 'pjac', 'gsensi', 'gspec', 'tsensi', 'tspec',]
assert(len(PLURAL_KEYS) == len(SINGULAR_KEYS))

Coef = namedtuple('Coef', SINGULAR_KEYS[1:])


def calc_coef(outputs, labels):
    assert(isinstance(labels, torch.Tensor))
    assert(isinstance(outputs, torch.Tensor))
    assert(outputs.size() == labels.size())
    dice, jac = similarity_index(outputs, labels)
    output_values = revert_onehot(outputs)
    label_values = revert_onehot(labels)
    pdice, pjac = pixel_similarity_index(output_values, label_values)
    output_glands = torch.gt(output_values, IDX_NONE)
    label_glands = torch.gt(label_values, IDX_NONE)
    output_tumors = torch.gt(output_values, IDX_NORMAL)
    label_tumors = torch.gt(label_values, IDX_NORMAL)
    gsensi, gspec = inspection_accuracy(output_glands, label_glands)
    tsensi, tspec = inspection_accuracy(output_tumors, label_tumors)
    return Coef(dice, jac, pdice, pjac, gsensi, gspec, tsensi, tspec)

def coef_to_str(coef):
    l = []
    for c, v in coef._asdict().items():
        l.append(f'{c}={v:.3f}')
    return ' '.join(l)

def safe_average(data):
    if len(data) > 0:
        return np.average(data)
    else:
        return 0.0

class Metrics():
    def __init__(self):
        self.data = OrderedDict()
        for key in PLURAL_KEYS:
            self.data[key]  = []

    def append_loss(self, loss):
        self.data['losses'].append(loss)

    def append_coef(self, coef):
        for i, s in enumerate(SINGULAR_KEYS):
            if i is 0:
                continue
            p = PLURAL_KEYS[i]
            self.data[p].append(getattr(coef, s))

    def append_values(self, loss, dice, jac, pdice, pjac, gsensi, gspec, tsensi, tspec):
        self.data['losses'].append(loss)
        self.data['dices'].append(dice)
        self.data['jacs'].append(jac)
        self.data['pdices'].append(pdice)
        self.data['pjacs'].append(pjac)
        self.data['gsensis'].append(gsensi)
        self.data['gspecs'].append(gspec)
        self.data['tsensis'].append(tsensi)
        self.data['tspecs'].append(tspec)

    def append_nested_metrics(self, nested):
        for key in PLURAL_KEYS:
            self.data[key] += nested.data[key]

    def load_state_dict(self, data):
        self.data = copy.deepcopy(data)

    def state_dict(self):
        return copy.deepcopy(self.data)

    def avg_loss(self):
        return safe_average(self.data['losses'])

    def last_loss(self):
        return self.data['losses'][-1]

    def avg_coef(self):
        l = []
        for i, p in enumerate(PLURAL_KEYS[1:]):
            l.append(safe_average(self.data[p]))
        return Coef(*l)

    def last_coef(self):
        l = []
        for i, p in enumerate(PLURAL_KEYS[1:]):
            l.append(self.data[p][-1])
        return Coef(*l)

    def avg(self, key):
        return safe_average(self.data[key])

    def last(self, key):
        return self.data[key][-1]

    def get(self, key):
        return self.data.get(key)
