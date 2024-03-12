import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.sp_dataset import SPDataset
from model.sp_transformer import TransformerClassifier
from params import ROOT_DIR, DATA, BATCH_SIZE, MODEL, DEVICE, EPOCHS


def __load_model(model):
    pass


def train():
    # LOAD DATASET
    train_set = SPDataset(
        json_paths=[str(Path(ROOT_DIR) / 'data/sp_data/train_set_partition_0.json'),
                    str(Path(ROOT_DIR) / 'data/sp_data/train_set_partition_1.json')],
        data_type=DATA
    )
    val_set = SPDataset(
        json_paths=[str(Path(ROOT_DIR) / 'data/sp_data/test_set_partition_0.json'),
                    str(Path(ROOT_DIR) / 'data/sp_data/test_set_partition_1.json')],
        data_type=DATA
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # LOAD MODEL
    config = json.load(open(str(Path(ROOT_DIR) / f'configs/{MODEL}_config_default.json')))
    model = TransformerClassifier(config).to(DEVICE)

    # OPTIM AND LOSS
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # TRAIN PROCESS
    for epoch in range(EPOCHS):
        print('Epoch {}/{}:'.format(epoch + 1, EPOCHS))
        model.train()
        if torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(
                backend='nccl', world_size=torch.cuda.device_count()
            )
            model = torch.nn.parallel.DistributedDataParallel(model)
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
        for _, batch in enumerate(val_loader):
            x, lb, kingdom = batch
            x = x.to(DEVICE)
            lb = lb.to(DEVICE)
            pred = model(x)
            val_outputs_lb.extend(lb)
            val_outputs_pred.extend(pred)

        all_lb = torch.argmax(torch.tensor(val_outputs_lb, device=DEVICE))
        all_pred = torch.argmax(torch.tensor(val_outputs_pred, device=DEVICE))
        print(classification_report(all_lb, all_pred))


if __name__ == '__main__':
    train()
