import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from SalesDataSet import SalesDataSet
from BaseModel import BaseModel


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred, actual))


if __name__ == '__main__':
    data_dir = '../final_draft/data/train_data.csv'
    TIME_WINDOW = 30

    train_set = SalesDataSet(data_dir, TIME_WINDOW, 'train')
    valid_set = SalesDataSet(data_dir, TIME_WINDOW, 'valid')

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=10, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    print(len(train_loader))
    model = BaseModel(train_set.embedding_sizes, emb_size=64, num_lstm_layers=4, hidden_dim=256, dropout=0.3, time_window=TIME_WINDOW)
    # model = BaseModel(train_set.embedding_sizes, emb_size=64, hidden_dim=128, dropout=0.3, att_dim=128,
    #                   time_window=TIME_WINDOW)
    # model = torch.load('./models/bert_days_30_21.pkl')
    # criterion = RMSLELoss()
    # criterion = nn.MSELoss()
    criterion = RMSELoss()
    optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
    bar = tqdm(total=25 * len(train_loader))
    for epoch in range(25):
        running_loss = 0.0
        model.train()
        for step, data in enumerate(train_loader):
            optimizer.zero_grad()
            X, y = data
            outputs = model(X)
            targets = y['target_sales'].to('cuda').float()
            loss = criterion(outputs.flatten(), targets.flatten())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            bar.update(1)

            if step % 1000 == 999:
                print('[%d, %5d] Training Loss: %.6f' % (epoch + 1, step + 1, (running_loss / 1000)))
                running_loss = 0.0

        with torch.no_grad():
            model.eval()
            eval_loss = 0.0
            batches = 0
            for e_step, e_data in enumerate(tqdm(valid_loader, position=0, leave=True)):
                batches += 1
                X, y = e_data
                outputs = model(X)
                targets = y['target_sales'].to('cuda').float()
                loss = criterion(outputs, targets)
                eval_loss += loss.item()
        scheduler.step()
        print('[Epoch %d] Eval_Loss: %.6f' % (epoch + 1, (eval_loss / batches)))
        torch.save(model, './models/lstm_days_{}_{}.pkl'.format(TIME_WINDOW, epoch))

