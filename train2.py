import json

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data.sp_dataset import SPDataset
from models.sp_transformer import TransformerClassifier
from params import DATA, BATCH_SIZE, MODEL, DEVICE, EPOCHS


def __load_model(model):
    pass


def train():
    # LOAD DATASET
    train_set = SPDataset(
        json_paths=utils.abspaths(
            ['data/sp_data/train_set_partition_0.json', 'data/sp_data/train_set_partition_1.json']),
        dtype=DATA
    )
    val_set = SPDataset(
        json_paths=utils.abspaths(['data/sp_data/test_set_partition_0.json', 'data/sp_data/test_set_partition_1.json']),
        dtype=DATA
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # LOAD MODEL
    config = json.load(open(utils.abspath(f'configs/model_configs/{MODEL}_config_default.json')))
    model = TransformerClassifier(config).to(DEVICE)

    # OPTIM AND LOSS
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # TRAIN PROCESS
    for epoch in range(EPOCHS):
        print('Epoch {}/{}:'.format(epoch + 1, EPOCHS))
        model.train()
        for _, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            x, lb, kingdom = batch
            x = x.to(DEVICE)
            lb = lb.to(DEVICE)
            # kingdom = kingdom.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred.float(), lb.float())
            loss.backward()
            optimizer.step()

        val_outputs_lb = []
        val_outputs_pred = []
        model.eval()
        for _, batch in enumerate(tqdm(val_loader)):
            x, lb, kingdom = batch
            x = x.to(DEVICE)
            lb = lb.to(DEVICE)
            pred = model(x)
            val_outputs_lb.append(lb)
            val_outputs_pred.append(pred)

        all_lb = torch.argmax(torch.cat(val_outputs_lb), dim=1)
        all_pred = torch.argmax(torch.cat(val_outputs_pred), dim=1)
        print(classification_report(all_lb, all_pred))


if __name__ == '__main__':
    train()
