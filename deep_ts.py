import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score

class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = F.relu(self.fc(out[:, -1, :]))
        out = F.relu(self.fc(out))
        out = self.fc_out(out)
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

def Trainer(model, trn_dl, val_dl, n_epochs, sched, opt, criterion, device, patience):
    print('Start model training')
    print(model)
    trials = 0
    best_acc = 0
    best_score = 0

    for epoch in range(1, n_epochs + 1):

        for i, (x_batch, y_batch) in enumerate(trn_dl):
            model.train()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            opt.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        correct, total = 0, 0
        total_preds, total_y = [], []
        for x_val, y_val in val_dl:
            x_val, y_val = [t.to(device) for t in (x_val, y_val)]
            out = model(x_val)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()

            total_preds.append(preds)
            total_y.append(y_val)

        acc = correct / total
        total_preds = np.concatenate([p.to("cpu") for p in total_preds])
        total_y = np.concatenate([y.to("cpu") for y in total_y])
        score = f1_score(total_y, total_preds, average="macro")

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. F1 score: {score:2.2%}, Acc.: {acc:2.2%}')

        if score > best_score:
            trials = 0
            best_score = score
            torch.save(model.state_dict(), 'best.pth')
            print(f'Epoch {epoch} best model saved with f1_score: {best_score:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break

    return best_score