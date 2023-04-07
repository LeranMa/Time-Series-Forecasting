# coding: utf-8
from torch.utils.data import Dataset
import itertools
import pandas as pd
import numpy as np
import math
import torch
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


class SalesDataSet(Dataset):

    def __init__(self, _sales_data_dir: str, _time_window: int = 30, _type='train'):
        super(SalesDataSet, self).__init__()
        self.sales_data_dir = _sales_data_dir
        self.time_window = _time_window
        self.type = _type
        self.data = pd.read_csv(self.sales_data_dir)

        self.data.sales = self.data.sales.map(lambda x: np.log(x + 1))
        self.data.dcoilwtico = self.data.dcoilwtico.map(lambda x: np.log(x + 1))
        self.data.transactions = self.data.transactions.map(lambda x: np.log(x + 1))
        self.data.onpromotion = self.data.onpromotion.map(lambda x: np.log(x + 1))

        self.family_dict = {family: int(i) for i, family in enumerate(sorted(self.data.family.unique()))}
        self.store_dict = {store: int(i) for i, store in enumerate(sorted(self.data.store_nbr.unique()))}
        self.month_dict = {month: int(i) for i, month in enumerate(sorted(self.data.month.unique()))}
        self.day_m_dict = {day: int(i) for i, day in enumerate(sorted(self.data.day_m.unique()))}
        self.city_dict = {city: int(i) for i, city in enumerate(sorted(self.data.city.unique()))}
        self.state_dict = {state: int(i) for i, state in enumerate(sorted(self.data.state.unique()))}
        self.type_dict = {t: int(i) for i, t in enumerate(sorted(self.data.type.unique()))}
        self.cluster_dict = {cluster: int(i) for i, cluster in enumerate(sorted(self.data.cluster.unique()))}
        self.holiday_dict = {holiday_type: int(i) for i, holiday_type in
                             enumerate(sorted(self.data.holiday_type.unique()))}

        self.embedding_sizes = {
            "family": [len(self.family_dict), math.ceil(math.sqrt(len(self.family_dict)))],
            "store": [len(self.store_dict), math.ceil(math.sqrt(len(self.store_dict)))],
            "month": [len(self.month_dict), math.ceil(math.sqrt(len(self.month_dict)))],
            "day": [7, math.ceil(math.sqrt(7))],
            "day_m": [len(self.day_m_dict), math.ceil(math.sqrt(len(self.day_m_dict)))],
            "city": [len(self.city_dict), math.ceil(math.sqrt(len(self.city_dict)))],
            "state": [len(self.state_dict), math.ceil(math.sqrt(len(self.state_dict)))],
            "type": [len(self.type_dict), math.ceil(math.sqrt(len(self.type_dict)))],
            "cluster": [len(self.cluster_dict), math.ceil(math.sqrt(len(self.cluster_dict)))],
            "holiday": [len(self.holiday_dict), math.ceil(math.sqrt(len(self.holiday_dict)))],
        }

        self.data.family = self.data.family.map(self.family_dict)
        self.data.store_nbr = self.data.store_nbr.map(self.store_dict)
        self.data.month = self.data.month.map(self.month_dict)
        self.data.day_m = self.data.day_m.map(self.day_m_dict)
        self.data.city = self.data.city.map(self.city_dict)
        self.data.state = self.data.state.map(self.state_dict)
        self.data.type = self.data.type.map(self.type_dict)
        self.data.cluster = self.data.cluster.map(self.cluster_dict)
        self.data.holiday_type = self.data.holiday_type.map(self.holiday_dict)

        self.stores = self.data.store_nbr.unique()
        self.families = self.data.family.unique()
        self.store_family = sorted(itertools.product(self.stores, self.families))
        self.timestamps = sorted(self.data.date.unique())
        self.time_pairs = []

        if self.type == 'train':
            self.date_range = self.timestamps[:-16]
            for i, day in enumerate(self.date_range):
                if i < self.time_window:
                    continue
                if i+16 > len(self.date_range):
                    break
                past = self.timestamps[i - self.time_window:i]
                future = self.timestamps[i:i+16]
                self.time_pairs.append([day, min(past), max(past), min(future), max(future)])
        else:
            self.date_range = self.timestamps[-16:]
            for i, day in enumerate(self.date_range):
                if i+16 > len(self.date_range):
                    break
                idx = self.timestamps.index(day)
                past = self.timestamps[idx - self.time_window:idx]
                future = self.timestamps[idx:idx+16]
                self.time_pairs.append([day, min(past), max(past), min(future), max(future)])

        # temp = []
        # for pair in self.time_pairs:
        #     if self.type == 'train':
        #         if pair[0] < self.timestamps[-(16 + self.time_window)]:
        #             temp.append(pair)
        #     else:
        #         if pair[0] >= self.timestamps[-(16 + self.time_window)]:
        #             temp.append(pair)
        # self.time_pairs = temp

        self.data_idx = sorted(itertools.product(self.store_family, self.time_pairs))
        self.data = self.data.set_index(['store_nbr', 'family']).sort_index()
        self.store_family_data = {key: self.data.loc[key].reset_index().set_index('date').sort_index()
                                  for key in self.store_family}

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        today = self.data_idx[idx][1][0]
        start_date = self.data_idx[idx][1][1]
        end_date = self.data_idx[idx][1][2]
        store_family_data = self.store_family_data[self.data_idx[idx][0]]
        today_data = store_family_data.loc[today]
        past_data = store_family_data.loc[start_date:end_date]
        future_data = store_family_data.loc[self.data_idx[idx][1][3]: self.data_idx[idx][1][4]]

        X = {
            "family": int(today_data['family']),
            "store_nbr": int(today_data['store_nbr']),
            "city": int(today_data['city']),
            "state": int(today_data['state']),
            "type": int(today_data['type']),
            "cluster": int(today_data['cluster']),

            "past_month": past_data["month"].values.astype(int),
            "past_day": past_data["day"].values.astype(int),
            "past_day_m": past_data["day_m"].values.astype(int),
            "past_holiday": past_data["holiday_type"].values.astype(int),
            'past_continues': past_data[["dcoilwtico", "sales", "onpromotion", "transactions"]].values,

            "future_month": future_data["month"].values.astype(int),
            "future_day": future_data["day"].values.astype(int),
            "future_day_m": future_data["day_m"].values.astype(int),
            "future_holiday": future_data["holiday_type"].values.astype(int),
            'future_continues': np.concatenate((future_data[["dcoilwtico", "onpromotion"]].values,
                                               # fourierExtrapolation(past_data.sales.values[-7:], 16)[-16:].reshape((-1, 1)),
                                               # fourierExtrapolation(past_data.transactions.values[-7:], 16)[-16:].reshape((-1, 1))
                                               ),
                                               axis=1),
        }

        y = {"target_sales": future_data["sales"].values.astype(np.float32)}

        return X, y


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split

    data_dir = '../final_draft//data/train_data.csv'
    train_set = SalesDataSet(data_dir, 30, 'train')
    valid_set = SalesDataSet(data_dir, 30, 'valid')
    print(len(train_set))
    print(len(valid_set))
    # print(len(dataset))
    #
    # len_data = len(dataset)
    # train_len = int(len_data * 0.9)
    # valid_len = len_data - train_len
    # train_set, valid_set = random_split(dataset, [train_len, valid_len], generator=torch.Generator().manual_seed(2021))
    # train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    # valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True)
    #
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

    for step, data in enumerate(train_loader):
        X, y = data
        print(X)
        print(y)
        break


    for step, data in enumerate(valid_loader):
        X, y = data
        print(X)
        print(y)
        break
