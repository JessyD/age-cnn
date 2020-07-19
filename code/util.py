""" Helper functions. """
import numpy as np
import pandas as pd
from monai import transforms
from monai.data import list_data_collate, PersistentDataset, Dataset
from torch.utils.data import DataLoader


def get_dataflow(seed, data_dir, cache_dir, batch_size):
    # Define transformations
    train_transforms = transforms.Compose([
        transforms.LoadNiftid(keys=["c1", "c2", "c3"]),
        transforms.AddChanneld(keys=["c1", "c2", "c3"]),
        transforms.ConcatItemsd(keys=["c1", "c2", "c3"], name="img"),
        transforms.ScaleIntensityd(keys="img"),
        transforms.ToTensord(keys=["img", "label"])
    ])

    val_transforms = transforms.Compose([
        transforms.LoadNiftid(keys=["c1", "c2", "c3"]),
        transforms.AddChanneld(keys=["c1", "c2", "c3"]),  # I suppose you label data already have 2 channels
        transforms.ConcatItemsd(keys=["c1", "c2", "c3"], name="img"),
        transforms.ScaleIntensityd(keys="img"),
        transforms.ToTensord(keys=["img", "label"])
    ])

    # Get img paths
    df = pd.read_csv(data_dir / "all_BANC_2019.csv")
    df = df.sample(frac=1, random_state=seed)
    df["NormAge"] = (((df["Age"] - 18) / (92 - 18)) * 2) - 1
    data_dicts = []
    for index, row in df.iterrows():
        study_dir = data_dir / row["Study"] / "derivatives" / "spm"
        subj = list(study_dir.glob(f"sub-{row['Subject']}"))

        if subj == []:
            subj = list(study_dir.glob(f"*sub-{row['Subject']}*"))
            if subj == []:
                subj = list(study_dir.glob(f"*sub-{row['Subject'].rstrip('_S1')}*"))
                if subj == []:
                    if row["Study"] == "SALD":
                        subj = list(study_dir.glob(f"sub-{int(row['Subject']):06d}*"))
                        if subj == []:
                            print(f"{row['Study']} {row['Subject']}")
                            continue


        c1_img = list(subj[0].glob("./smwc1*"))
        c2_img = list(subj[0].glob("./smwc2*"))
        c3_img = list(subj[0].glob("./smwc3*"))

        if c1_img == []:
            print(f"{row['Study']} {row['Subject']}")
            continue
        if c2_img == []:
            print(f"{row['Study']} {row['Subject']}")
            continue
        if c3_img == []:
            print(f"{row['Study']} {row['Subject']}")
            continue

        data_dicts.append({
            "c1": str(c1_img[0]),
            "c2": str(c2_img[0]),
            "c3": str(c3_img[0]),
            "label": row["NormAge"]
        })

    val_size = len(data_dicts) // 10
    # Create datasets and dataloaders
    # train_ds = PersistentDataset(data=data_dicts[:-val_size], transform=train_transforms,
    #                              cache_dir=cache_dir)
    train_ds = Dataset(data=data_dicts[:-val_size], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=list_data_collate)

    # val_ds = PersistentDataset(data=data_dicts[-val_size:], transform=val_transforms,
    #                            cache_dir=cache_dir)
    val_ds = Dataset(data=data_dicts[-val_size:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader
