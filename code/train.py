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
from sklearn.model_selection import train_test_split


def get_data(train_path, df):
    # Load dataset - 50 for the moment
    data = {'subject_id': [], 'imgs': [], 'labels': []}
    for index, row in df.iloc[:50].iterrows():
        path = train_path / row['Study'] /'derivatives' / 'spm' / str('sub-' +
                                                                      row['Subject'])
        nifti = path.glob('smwc1*.nii')
        img = str(next(nifti))
        # Mask the image to have only brain related information
        img = nib.load(img).get_data()
        img = np.nan_to_num(img)
        data['imgs'].append(img)
        data['subject_id'].append(row['Subject'])
        data['labels'].append(df[df['Subject'] == row['Subject']]['Age'].values[0])
    return data


PROJECT_ROOT = Path('/regeage')
data_path = PROJECT_ROOT / 'data' / 'BANC_2019'
train_path = data_path / 'train_data'
demographics_path = data_path / 'all_BANC_2019.csv'
df = pd.read_csv(data_path / 'cleaned_BANC_2019.csv')
rnd_seed = 1234
NUM_CHANNELS = 1

data = get_data(train_path, df)
test_size = .6

X_train, X_test, y_train, y_test = train_test_split(data['imgs'],
                                                    data['labels'],
                                                    test_size=test_size,
                                                    random_state=rnd_seed)

import pdb
pdb.set_trace()

