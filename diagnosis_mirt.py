import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from model import MIRTNet
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

train_data = pd.read_csv("data/train.csv")
valid_data = pd.read_csv("data/valid.csv")
test_data = pd.read_csv("data/test.csv")

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

cd_model = MIRTNet(4164, 17747, 100, None)
loss_function = nn.BCELoss()

trainer = torch.optim.Adam(cd_model.parameters(), lr)

for e in range(epoch):
    losses = []
    for batch_data in tqdm(train, "Epoch %s" % e):
        user_id, item_id, response = batch_data
        # user_id: torch.Tensor = user_id.cuda()
        # item_id: torch.Tensor = item_id.cuda()
        predicted_response: torch.Tensor = cd_model(user_id, item_id)
        # response: torch.Tensor = response.cuda()
        loss = loss_function(predicted_response, response)

        # back propagation
        trainer.zero_grad()
        loss.backward()
        trainer.step()

        losses.append(loss.mean().item())
    print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

    if test_data is not None:
        cd_model.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test, "evaluating"):
            user_id, item_id, response = batch_data
            # user_id: torch.Tensor = user_id.cuda()
            # item_id: torch.Tensor = item_id.cuda()
            pred: torch.Tensor = cd_model(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        cd_model.train()
        print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)))
