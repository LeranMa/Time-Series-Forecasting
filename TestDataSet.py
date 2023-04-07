# coding: utf-8
from torch.utils.data import Dataset
import itertools
import pandas as pd
import numpy as np
import math
import torch
from SalesDataSet import SalesDataSet
from numpy import fft

def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = 15  # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)  # find linear trend in x
    # x_notrend = x - p[0] * t  # detrended x
    x_notrend = x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)  # frequencies
    indexes = list(range(n))
    # sort indexes by power specturm, higher -> lower
    indexes.sort(key=lambda i: x_freqdom[i] * np.conj(x_freqdom[i]) / n, reverse=True)

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n  # amplitude
        phase = np.angle(x_freqdom[i])  # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    # return restored_sig + p[0] * t
    return restored_sig

class TestDataSet(Dataset):

    def __init__(self, future_data, past_data, today_data):
        super(TestDataSet, self).__init__()
        self.future_data = future_data
        self.past_data = past_data
        self.today_data = today_data

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        X = {
            "family": int(self.today_data['family']),
            "store_nbr": int(self.today_data['store_nbr']),
            "city": int(self.today_data['city']),
            "state": int(self.today_data['state']),
            "type": int(self.today_data['type']),
            "cluster": int(self.today_data['cluster']),

            "past_month": self.past_data["month"].values.astype(int),
            "past_day": self.past_data["day"].values.astype(int),
            "past_day_m": self.past_data["day_m"].values.astype(int),
            "past_holiday": self.past_data["holiday_type"].values.astype(int),
            'past_continues': self.past_data[["dcoilwtico", "sales", "onpromotion", "transactions"]].values,

            "future_month": self.future_data["month"].values.astype(int),
            "future_day": self.future_data["day"].values.astype(int),
            "future_day_m": self.future_data["day_m"].values.astype(int),
            "future_holiday": self.future_data["holiday_type"].values.astype(int),
            'future_continues': np.concatenate((self.future_data[["dcoilwtico", "onpromotion"]].values,
                                               # fourierExtrapolation(self.past_data.sales.values[-7:], 16)[-16:].reshape((-1, 1)),
                                               # fourierExtrapolation(self.past_data.transactions.values[-7:], 16)[-16:].reshape((-1, 1))
                                               ),
                                               axis=1),
        }
        return X
        
