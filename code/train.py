#!/usr/bin/env python3
"""
This code trains the model
"""
from pathlib import Path

import numpy as np
import torch
from monai.utils import set_determinism
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import DeepNet
from util import get_dataflow

seed = 0
set_determinism(seed=seed)
np.random.seed(seed)

data_dir = Path("/project/data/BANC/")
output_dir = Path("/project/outputs/")
output_dir.mkdir(exist_ok=True)

batch_size = 2
cache_dir = output_dir / "cached_data"
cache_dir.mkdir(exist_ok=True)
train_loader, val_loader = get_dataflow(seed, data_dir, cache_dir, batch_size)

# Create DenseNet121, CrossEntropyLoss and Adam optimizer
device = torch.device("cuda")
model = DeepNet()
model = model.to(device)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)

# start a typical PyTorch training
n_epochs = 1000
val_interval = 2
best_metric = 10000
# writer_train = SummaryWriter()
# writer_val = SummaryWriter()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_loader.dataset) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        # writer_train.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = val_data["img"].to(device), val_data[
                    "label"].to(device)
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)

            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
            if acc_metric > best_metric:
                best_metric = acc_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, acc_metric, auc_metric, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()