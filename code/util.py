""" Helper functions. """
import nibabel as nib
import numpy as np
import pandas as pd
from monai import transforms
from monai.data import list_data_collate, PersistentDataset, Dataset
from torch.utils.data import DataLoader


def get_dataflow(seed, data_dir, cache_dir, batch_size):
    img = nib.load(str(data_dir / "average_smwc1.nii"))
    img_data_1 = img.get_fdata()
    img_data_1 = np.expand_dims(img_data_1, axis=0)

    img = nib.load(str(data_dir / "average_smwc2.nii"))
    img_data_2 = img.get_fdata()
    img_data_2 = np.expand_dims(img_data_2, axis=0)

    img = nib.load(str(data_dir / "average_smwc3.nii"))
    img_data_3 = img.get_fdata()
    img_data_3 = np.expand_dims(img_data_3, axis=0)

    mask = np.concatenate((img_data_1, img_data_2, img_data_3))
    mask[mask > 0.3] = 1
    mask[mask <= 0.3] = 0

    # Define transformations
    train_transforms = transforms.Compose([
        transforms.LoadNiftid(keys=["c1", "c2", "c3"]),
        transforms.AddChanneld(keys=["c1", "c2", "c3"]),
        transforms.ConcatItemsd(keys=["c1", "c2", "c3"], name="img"),
        transforms.MaskIntensityd(keys=["img"], mask_data=mask),
        transforms.ScaleIntensityd(keys="img"),
        transforms.ToTensord(keys=["img", "label"])
    ])

    val_transforms = transforms.Compose([
        transforms.LoadNiftid(keys=["c1", "c2", "c3"]),
        transforms.AddChanneld(keys=["c1", "c2", "c3"]),
        transforms.ConcatItemsd(keys=["c1", "c2", "c3"], name="img"),
        transforms.MaskIntensityd(keys=["img"], mask_data=mask),
        transforms.ScaleIntensityd(keys="img"),
        transforms.ToTensord(keys=["img", "label"])
    ])

    # Get img paths
    df = pd.read_csv(data_dir / "banc2019_training_dataset.csv")
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
                    else:
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

    print(f"Found {len(data_dicts)} subjects.")
    val_size = len(data_dicts) // 10
    # Create datasets and dataloaders
    train_ds = PersistentDataset(data=data_dicts[:-val_size], transform=train_transforms,
                                 cache_dir=cache_dir)
    # train_ds = Dataset(data=data_dicts[:-val_size], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=list_data_collate)

    val_ds = PersistentDataset(data=data_dicts[-val_size:], transform=val_transforms,
                               cache_dir=cache_dir)
    # val_ds = Dataset(data=data_dicts[-val_size:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader


def get_test_banc_dataflow(seed, data_dir, batch_size):
    img = nib.load(str(data_dir / "average_smwc1.nii"))
    img_data_1 = img.get_fdata()
    img_data_1 = np.expand_dims(img_data_1, axis=0)

    img = nib.load(str(data_dir / "average_smwc2.nii"))
    img_data_2 = img.get_fdata()
    img_data_2 = np.expand_dims(img_data_2, axis=0)

    img = nib.load(str(data_dir / "average_smwc3.nii"))
    img_data_3 = img.get_fdata()
    img_data_3 = np.expand_dims(img_data_3, axis=0)

    mask = np.concatenate((img_data_1, img_data_2, img_data_3))
    mask[mask > 0.3] = 1
    mask[mask <= 0.3] = 0

    # Define transformations
    test_transforms = transforms.Compose([
        transforms.LoadNiftid(keys=["c1", "c2", "c3"]),
        transforms.AddChanneld(keys=["c1", "c2", "c3"]),
        transforms.ConcatItemsd(keys=["c1", "c2", "c3"], name="img"),
        transforms.MaskIntensityd(keys=["img"], mask_data=mask),
        transforms.ScaleIntensityd(keys="img"),
        transforms.ToTensord(keys=["img", "label"])
    ])

    # Get img paths
    df = pd.read_csv(data_dir / "banc2019_testing_dataset.csv")
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
                    else:
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
            "study": row['Study'],
            "subj": row['Subject'],
            "age": row['Age'],
            "c1": str(c1_img[0]),
            "c2": str(c2_img[0]),
            "c3": str(c3_img[0]),
            "label": row["NormAge"]
        })

    print(f"Found {len(data_dicts)} subjects.")
    # Create datasets and dataloaders
    test_ds = Dataset(data=data_dicts, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             num_workers=4)

    return test_loader
