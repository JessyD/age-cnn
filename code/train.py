#!/usr/bin/env python3
"""
This code trains the model
"""
from monai.utils import set_determinism
import numpy as np
from model import DeepNet
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch import nn
import monai
from monai.transforms import Compose, LoadNiftid, AddChanneld, ScaleIntensityd, Resized, RandRotate90d, ToTensord

seed = 0
set_determinism(seed=seed)
np.random.seed(seed)

# 2 binary labels for gender classification: man and woman
labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  dtype=np.int64)
train_files = [{"img": img, "label": label} for img, label in
               zip(images[:10], labels[:10])]
val_files = [{"img": img, "label": label} for img, label in
             zip(images[-10:], labels[-10:])]

# Define transforms for image
train_transforms = Compose(
    [
        LoadNiftid(keys=["img"]),
        AddChanneld(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(96, 96, 96)),
        RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
        ToTensord(keys=["img"]),
    ]
)
val_transforms = Compose(
    [
        LoadNiftid(keys=["img"]),
        AddChanneld(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(96, 96, 96)),
        ToTensord(keys=["img"]),
    ]
)

# create a training data loader
train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4,
                          pin_memory=torch.cuda.is_available())

# create a validation data loader
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=4,
                        pin_memory=torch.cuda.is_available())

# Create DenseNet121, CrossEntropyLoss and Adam optimizer
device = torch.device("cuda:0")
model = DeepNet()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)

# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
writer = SummaryWriter()
for epoch in range(5):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{5}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
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