import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from model import MIRTNet, MFNet
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

train_data = pd.read_csv("../../data/a0910/train.csv")
valid_data = pd.read_csv("../../data/a0910/valid.csv")
test_data = pd.read_csv("../../data/a0910/test.csv")

batch_size = 256
epoch = 5
lr = 0.01


def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, valid, test = [
    transform(data["user_id"], data["item_id"], data["score"], batch_size)
    for data in [train_data, valid_data, test_data]
]


class HeckmanIRT(nn.Module):
    def __init__(self):
        super(HeckmanIRT, self).__init__()
        self.irt_net = MIRTNet(4164, 17747, 123, None).cuda()
        self.rec = torch.load('ncf_mlp.pth').cuda()
        self.reg1 = nn.Linear(2, 4).cuda()
        self.reg2 = nn.Linear(4, 1).cuda()
        for param in self.rec.parameters():
            param.requires_grad = False

    def forward(self, user, item):
        r1: torch.Tensor = self.irt_net(user, item)
        r2: torch.Tensor = F.sigmoid(self.rec(user, item))
        rf = torch.stack([r1, r2], dim=1)
        return F.sigmoid(self.reg2(F.relu(self.reg1(rf)))).squeeze()


hmodel = HeckmanIRT()
# hmodel = MIRTNet(4164, 17747, 123, None).cuda()
loss_function = nn.BCELoss()

trainer = torch.optim.Adam(hmodel.parameters(), lr)

for e in range(epoch):
    losses = []
    for batch_data in tqdm(train, "Epoch %s" % e):
        user_id, item_id, response = batch_data
        user_id: torch.Tensor = user_id.cuda()
        item_id: torch.Tensor = item_id.cuda()
        predicted_response: torch.Tensor = hmodel(user_id, item_id)
        response: torch.Tensor = response.cuda()
        loss = loss_function(predicted_response, response)

        # back propagation
        trainer.zero_grad()
        loss.backward()
        trainer.step()

        losses.append(loss.mean().item())
    print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

    if test_data is not None:
        hmodel.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test, "evaluating"):
            user_id, item_id, response = batch_data
            user_id: torch.Tensor = user_id.cuda()
            item_id: torch.Tensor = item_id.cuda()
            pred: torch.Tensor = hmodel(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        hmodel.train()
        print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)))

