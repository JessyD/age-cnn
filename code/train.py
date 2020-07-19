#!/usr/bin/env python3
"""
This code trains the model
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from monai.utils import set_determinism
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import DeepNet
from util import get_dataflow

seed = 0
set_determinism(seed=seed)
np.random.seed(seed)

data_dir = Path("/project/data/BANC/")
output_dir = Path("/project/outputs/")
output_dir.mkdir(exist_ok=True)

batch_size = 50
cache_dir = output_dir / "cached_data"
cache_dir.mkdir(exist_ok=True)
train_loader, val_loader = get_dataflow(seed, data_dir, cache_dir, batch_size)

device = torch.device("cuda")
model = DeepNet()
model = model.to(device)
loss_func = torch.nn.MSELoss()

lr_gamma = 0.9999
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

# start a typical PyTorch training
n_epochs = 1000
val_interval = 1
best_metric = 10000
best_metric_epoch = 0

datenow = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = output_dir / datenow
run_dir.mkdir(exist_ok=True)
writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    step = 0
    with tqdm(total=len(train_loader), desc=f"epoch {epoch}/{n_epochs}") as pbar:
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_len = len(train_loader.dataset) // train_loader.batch_size
            writer_train.add_scalar("loss", loss.item(), epoch_len * epoch + step)

            pbar.set_postfix({"loss": f"{loss.item():.6}"})
            pbar.update()

        epoch_loss /= step
        pbar.set_postfix({"loss": f"{epoch_loss:.6}"})
        pbar.update()

        if (epoch + 1) % val_interval == 0:
            epoch_val_loss = 0
            epoch_val_loss_y = 0
            val_step = 0
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_step += 1
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = model(val_images)
                    loss_val = loss_func(y_pred, val_labels)

                    mae = torch.nn.functional.l1_loss( ((((val_labels+1)/2)*(92-18))+18), ((((y_pred+1)/2)*(92-18))+18))

                    epoch_val_loss += loss_val.item()
                    epoch_val_loss += mae.item()

                epoch_val_loss /= val_step
                epoch_val_loss_y /= val_step
                pbar.set_postfix({"loss": f"{epoch_loss:.6}", "val_loss": f"{epoch_val_loss:.6}", "val_loss_y": f"{epoch_val_loss_y:.6}"})
                pbar.update()

                if epoch_val_loss < best_metric:
                    best_metric = epoch_val_loss
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), output_dir / "best_metric_model.pth")
                    print("saved new best metric model")
                writer_val.add_scalar("loss", epoch_val_loss, epoch_len * (epoch + 1))

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer_val.close()
writer_train.close()
