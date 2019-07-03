import copy
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

KEYS = ['losses', 'dices', 'jacs', 'pdices', 'pjacs', 'gsensis', 'gspecs', 'tsensis', 'tspecs',]

class Metrics():
    def __init__(self):
        self.data = {}
        for key in KEYS:
            self.data[key]  = []

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

    def get_avg_values(self):
        return [np.average(self.data[key]) for key in KEYS]

    def append_metrics(self, metrics):
        self.append_values(*metrics.get_avg_values())

    def load_state_dict(self, data):
        self.data = copy.deepcopy(data)

    def state_dict(self):
        return copy.deepcopy(self.data)

    def avg(self, key):
        return np.average(self.data[key])

    def last(self, key):
        return self.data[key][-1]

    def get(self, key):
        return self.data.get(key)
