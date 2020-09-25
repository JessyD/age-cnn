#!/usr/bin/env python3
"""
This code predict the BANC test set.
"""
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from monai.utils import set_determinism
from tqdm import tqdm

from util import get_test_banc_dataflow

seed = 0
set_determinism(seed=seed)
np.random.seed(seed)

data_dir = Path("/project/data/BANC/")
output_dir = Path("/project/outputs/")
output_dir.mkdir(exist_ok=True)

batch_size = 50
test_loader = get_test_banc_dataflow(seed, data_dir, batch_size)

device = torch.device("cuda")
model = torch.load(output_dir / "best_metric_model.pth")
model.eval()
model = model.to(device)

df = pd.DataFrame()

test_loss_y = 0
test_step = 0
with torch.no_grad():
    with tqdm(total=len(test_loader)) as pbar:
        for test_data in test_loader:
            test_step += 1
            test_images, test_labels = test_data["img"].to(device), test_data["label"].to(device)
            y_pred = model(test_images)
            mae = torch.nn.functional.l1_loss(((((test_labels + 1) / 2) * (92 - 18)) + 18),
                                              ((((y_pred + 1) / 2) * (92 - 18)) + 18))

            test_loss_y += mae.item()

            for subj, study, age, pred_age in zip(test_data["subj"], test_data["study"], test_data["age"],
                                                  ((((y_pred + 1) / 2) * (92 - 18)) + 18).cpu()):
                df = df.append(
                    {
                        "study": study,
                        "subj": subj,
                        "age": age.item(),
                        "pred_age": pred_age.item(),
                    }, ignore_index=True)

            pbar.update()

        test_loss_y /= test_step
        print(f"Test MAE: {test_loss_y:.6}")

df.to_csv(output_dir / "BANC_test.csv", sep="\t", index=False)