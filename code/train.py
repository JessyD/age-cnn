#!/usr/bin/env python3
"""
This code trains the model
"""
from pathlib import Path
import glob

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.masking import apply_mask


PROJECT_ROOT = Path('/regeage')
data_path = PROJECT_ROOT / 'data' / 'BANC_2019'
train_path = data_path / 'train_data'
demographics_path = data_path / 'all_BANC_2019.csv'
df = pd.read_csv(data_path / 'cleaned_BANC_2019.csv')
# Load the mask image
brain_mask = data_path / 'MNI152_T1_1.5mm_brain_masked2.nii.gz'
mask_img = nib.load(str(brain_mask))

# Load dataset - 100 for the moment
imgs = []
for index, row in df.iloc[:100].iterrows():
    path = train_path / row['Study'] /'derivatives' / 'spm' / str('sub-' +
                                                                  row['Subject'])
    nifti = path.glob('smwc1*.nii')
    img = str(next(nifti))
    # Mask the image to have only brain related information
    img = apply_mask(img, mask_img)
    img = np.asarray(img, dtype='float64')
    img = np.nan_to_num(img)
    imgs.append(img)

    import pdb
    pdb.set_trace()


