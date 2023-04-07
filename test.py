# coding: utf-8
import pandas as pd
import numpy as np
import torch
from SalesDataSet import SalesDataSet
from tqdm import tqdm
from TestDataSet import TestDataSet
from torch.utils.data import DataLoader, random_split
from datetime import datetime, timedelta

train_dir = '../final_draft/data/train_data.csv'
test_dir = '../final_draft/data/test_data.csv'
TIME_WINDOW = 30
train_data = SalesDataSet(train_dir, _time_window=TIME_WINDOW)
test_data = pd.read_csv(test_dir)
test_data.family = test_data.family.map(train_data.family_dict)
test_data.store_nbr = test_data.store_nbr.map(train_data.store_dict)
test_data.month = test_data.month.map(train_data.month_dict)
test_data.day_m = test_data.day_m.map(train_data.day_m_dict)
test_data.city = test_data.city.map(train_data.city_dict)
test_data.state = test_data.state.map(train_data.state_dict)
test_data.type = test_data.type.map(train_data.type_dict)
test_data.cluster = test_data.cluster.map(train_data.cluster_dict)
test_data.holiday_type = test_data.holiday_type.map(train_data.holiday_dict)
test_data.dcoilwtico = test_data.dcoilwtico.map(lambda x: np.log(x + 1))
test_data.transactions = test_data.transactions.map(lambda x: np.log(x + 1))
test_data.onpromotion = test_data.onpromotion.map(lambda x: np.log(x + 1))
_test_data = test_data[(test_data.date >= '2017-08-16')]
_today_data = test_data[(test_data.date == '2017-08-16')]

date = str(datetime(2017, 8, 15, 0, 0) - timedelta(TIME_WINDOW))[:10]

data = train_data.data[train_data.data.date > date]
pairs = train_data.store_family
model = torch.load('./models/lstm_days_{}_{}.pkl'.format(TIME_WINDOW, 3))
model.eval()

results = {}
for pair in tqdm(pairs):
    past_data = data.loc[pair].reset_index()
    future_data = _test_data[(_test_data.store_nbr == pair[0]) & (_test_data.family == pair[1])].reset_index()
    today_data = _today_data[(_today_data.store_nbr == pair[0]) & (_today_data.family == pair[1])].reset_index().iloc[0]
    t = TestDataSet(future_data, past_data, today_data)
    X = list(DataLoader(t, batch_size=2))[0]
    with torch.no_grad():
        model.eval()
        prediction = model(X).cpu().numpy()[0]
        _ids = list(test_data[(test_data.store_nbr == pair[0]) & (test_data.family == pair[1])].id)
        results.update(dict(zip(_ids, prediction)))

submission = pd.DataFrame()
submission['id'] = list(results.keys())
submission['sales'] = list(results.values())
submission = submission.sort_values(by='id')
submission.sales = submission.sales.map(lambda x: max(0, np.exp(x) - 1))
submission.to_csv('submission.csv', index=False)
