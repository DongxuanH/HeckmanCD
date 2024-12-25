import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import data_utils
from model import NCF

epochs = 10
num_ng = 2
batch_size = 32
factor_num = 32
num_layers = 3
dropout = 0.
archi = 'MLP'
lr = 0.001

train_data, user_num, item_num, train_mat = data_utils.load_data("data/train.csv")
valid_data, _, _, valid_mat = data_utils.load_data("data/valid.csv")

train_dataset = data_utils.NCFData(train_data, item_num, train_mat, num_ng, True)
valid_dataset = data_utils.NCFData(valid_data, item_num, valid_mat, 0, True)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

val_sum = len(valid_dataset)

GMF_model = None
MLP_model = None

model = NCF(user_num, item_num, factor_num, num_layers, dropout, archi, GMF_model, MLP_model)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_acc = 0.
for epoch in range(epochs):
    model.train()
    train_loader.dataset.ng_sample()

    count = 0
    for user, item, label in tqdm(train_loader):
        # user = user.cuda()
        # item = item.cuda()
        # label = label.float().cuda()
        model.zero_grad()
        prediction = model(user, item)
        loss = loss_function(prediction, label.float())
        loss.backward()
        optimizer.step()
        count += 1
        if count % 1000 == 0:
            print(f'Training Loss: {loss.item()}')

    model.eval()
    valid_loader.dataset.ng_sample()
    val_acc = 0
    for user, item, label in tqdm(valid_loader):
        # user = user.cuda()
        # item = item.cuda()
        # label = label.float().cuda()
        prediction = nn.Sigmoid()(model(user, item))
        val_acc += nn.Sigmoid()(model(user, item)).sum().item()
    val_acc /= val_sum
    print(val_acc)
    if val_acc > best_acc:
        best_acc, best_epoch = val_acc, epoch
        torch.save(model, f'ncf_mlp.pth')
