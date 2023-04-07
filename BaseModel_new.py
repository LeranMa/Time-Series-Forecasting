# coding: utf-8
import torch as tc
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, embedding_sizes, emb_size=32, hidden_dim=64, att_dim=64, dropout=0.3, time_window=16):
        super(BaseModel, self).__init__()
        self.family_emb_layer = tc.nn.Embedding(embedding_sizes['family'][0], emb_size)
        self.store_emb_layer = tc.nn.Embedding(embedding_sizes['store'][0], emb_size)
        self.month_emb_layer = tc.nn.Embedding(embedding_sizes['month'][0], emb_size)
        self.day_emb_layer = tc.nn.Embedding(embedding_sizes['day'][0], emb_size)
        self.city_emb_layer = tc.nn.Embedding(embedding_sizes['city'][0], emb_size)
        self.state_emb_layer = tc.nn.Embedding(embedding_sizes['state'][0], emb_size)
        self.type_emb_layer = tc.nn.Embedding(embedding_sizes['type'][0], emb_size)
        self.cluster_emb_layer = tc.nn.Embedding(embedding_sizes['cluster'][0], emb_size)
        self.holiday_emb_layer = tc.nn.Embedding(embedding_sizes['holiday'][0], emb_size)

        # self.today_linear = nn.Sequential(nn.Dropout(0.), nn.Linear(emb_size * 3 + 2, hidden_dim))
        self.past_continue_linear = nn.Sequential(nn.Dropout(0.), nn.Linear(emb_size * 3 + 4, hidden_dim))
        self.future_continue_linear = nn.Sequential(nn.Dropout(0.), nn.Linear(emb_size * 3 + 2, hidden_dim))
        self.stable_linear = nn.Sequential(nn.Dropout(0.), nn.Linear(emb_size * 6, hidden_dim))
        #self.past_lstm = nn.LSTM(hidden_dim, 2 * hidden_dim, num_layers=num_lstm_layers, batch_first=True, dropout=0.)
        #self.future_lstm = nn.LSTM(hidden_dim, 2 * hidden_dim, num_layers=num_lstm_layers, batch_first=True, dropout=0.)
        #self.bigger_stable = tc.nn.Linear(hidden_dim, 2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)


        self.bert = tc.nn.Sequential(nn.TransformerEncoderLayer(hidden_dim, 32, 2 * hidden_dim, batch_first=True, dropout=0.1),
                                     nn.TransformerEncoderLayer(hidden_dim, 32, 2 * hidden_dim, batch_first=True, dropout=0.1),
                                     nn.TransformerEncoderLayer(hidden_dim, 32, 2 * hidden_dim, batch_first=True, dropout=0.1),
                                     nn.TransformerEncoderLayer(hidden_dim, 16, 2 * hidden_dim, batch_first=True, dropout=0.1),
                                     nn.TransformerEncoderLayer(hidden_dim, 16, 2 * hidden_dim, batch_first=True, dropout=0.1),
                                     nn.TransformerEncoderLayer(hidden_dim, 16, 2 * hidden_dim, batch_first=True, dropout=0.1),
                                     nn.TransformerEncoderLayer(hidden_dim, 8, 2 * hidden_dim, batch_first=True, dropout=0.1),
                                     nn.TransformerEncoderLayer(hidden_dim, 8, 2 * hidden_dim, batch_first=True, dropout=0.1),
                                     nn.TransformerEncoderLayer(hidden_dim, 8, 2 * hidden_dim, batch_first=True, dropout=0.1),
                                     )
        self.embs = nn.Parameter(nn.init.xavier_normal_(tc.zeros(3, hidden_dim, device="cuda")), True)
        

        self.q_past = tc.nn.Linear(2*hidden_dim, att_dim)
        self.k_past = tc.nn.Linear(2*hidden_dim, att_dim)
        self.q_future = tc.nn.Linear(2*hidden_dim, att_dim)
        self.k_future = tc.nn.Linear(2*hidden_dim, att_dim)
        self.simple_predictor = tc.nn.Sequential(tc.nn.Dropout(0.1),
                                                 tc.nn.Linear(hidden_dim, 2*hidden_dim),
                                                 tc.nn.LeakyReLU(),
                                                 tc.nn.Dropout(0.25),
                                                 tc.nn.Linear(2*hidden_dim, 16))

        self.past_time_weights_store = tc.nn.Parameter(tc.tensor([[1.0 / i for i in range(time_window+1, 1, -1)]] * 54))
        self.past_time_weights_family = tc.nn.Parameter(tc.tensor([[1.0 / i ** 2 for i in range(time_window+1, 1, -1)]] * 33))
        self.future_time_weights_store = tc.nn.Parameter(tc.tensor([[1.0 / i for i in range(17, 1, -1)]] * 54))
        self.future_time_weights_family = tc.nn.Parameter(tc.tensor([[1.0 / i ** 2 for i in range(17, 1, -1)]] * 33))
        self.random_store = tc.nn.Parameter(tc.tensor([1.5] * 54))
        self.random_family = tc.nn.Parameter(tc.tensor([1.5] * 33))

        self.to('cuda')

    def forward(self, x):
        family = x['family'].to('cuda')
        store_nbr = x['store_nbr'].to('cuda')
        city = x['city'].to('cuda')
        state = x['state'].to('cuda')
        store_type = x['type'].to('cuda')
        cluster = x['cluster'].to('cuda')

        past_month = x['past_month'].to('cuda')
        past_day = x['past_day'].to('cuda')
        past_holiday = x['past_holiday'].to('cuda')
        past_continues = x['past_continues'].to('cuda')

        future_month = x['future_month'].to('cuda')
        future_day = x['future_day'].to('cuda')
        future_holiday = x['future_holiday'].to('cuda')
        future_continues = x['future_continues'].to('cuda')

        # today_month = x['today_month'].to('cuda')
        # today_day = x['today_day'].to('cuda')
        # today_holiday = x['today_holiday'].to('cuda')
        # today_oil = x['today_oil'].reshape((-1, 1)).to('cuda')
        # today_prom = x['today_prom'].reshape((-1, 1)).to('cuda')

        family_emb = self.family_emb_layer(family)  # (bs, dim)
        store_emb = self.store_emb_layer(store_nbr)
        city_emb = self.city_emb_layer(city)
        state_emb = self.state_emb_layer(state)
        type_emb = self.type_emb_layer(store_type)
        cluster_emb = self.cluster_emb_layer(cluster)

        past_month_emb = self.month_emb_layer(past_month)
        past_day_emb = self.day_emb_layer(past_day)
        past_holiday_emb = self.holiday_emb_layer(past_holiday)

        future_month_emb = self.month_emb_layer(future_month)
        future_day_emb = self.day_emb_layer(future_day)
        future_holiday_emb = self.holiday_emb_layer(future_holiday)

        stable = tc.cat([family_emb, store_emb, city_emb, state_emb, type_emb, cluster_emb], -1)  # (bs, dim)
        history = tc.cat([past_month_emb, past_day_emb, past_holiday_emb, past_continues], -1)
        future = tc.cat([future_month_emb, future_day_emb, future_holiday_emb, future_continues], -1)
        stables = self.stable_linear(stable.float())
        past_continues = self.past_continue_linear(history.float())

        future_continues = self.future_continue_linear(future.float())

        dynamic = self.bert(tc.cat([stables.unsqueeze(1) + self.embs[0],
                                    past_continues + self.embs[1],
                                    future_continues + self.embs[2]], 1).float())
        return self.simple_predictor(dynamic[:, 0])




if __name__ == '__main__':
    from SalesDataSet import SalesDataSet
    import torch
    from torch.utils.data import DataLoader, random_split


    class RMSELoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.mse = nn.MSELoss()

        def forward(self, pred, actual):
            return torch.sqrt(self.mse(pred, actual))


    data_dir = './data/train_data.csv'
    dataset = SalesDataSet(data_dir, _time_window=15)

    len_data = len(dataset)
    train_len = int(len_data * 0.9)
    valid_len = len_data - train_len
    train_set, valid_set = random_split(dataset, [train_len, valid_len], generator=torch.Generator().manual_seed(2021))
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=True)

    # model = torch.load('./models/first_model_0.pkl')
    model = BaseModel(dataset.embedding_sizes, hidden_dim=128)
    criterion = RMSELoss()
    for step, data in enumerate(train_loader):
        model.train()
        X, y = data

        print(criterion(model(X), y['target_sales'].cuda()))
        break
