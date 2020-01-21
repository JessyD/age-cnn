#!/usr/bin/env python3
"""
This code trains the model
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, optimizers
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
        img = nib.load(img).get_fdata()
        img = np.asarray(img, dtype='float32')
        img = np.nan_to_num(img)
        data['imgs'].append(img)
        data['subject_id'].append(row['Subject'])
        data['labels'].append(df[df['Subject'] == row['Subject']]['Age'].values[0])
    return data

def cnn_model():
    # Define parameters for the network
    filters = 1
    kernel_size = 3
    STRIDES_CONV = 1
    STRIDES_MAXPOOL = 2
    d1 = 121
    d2 = 145
    d3 = 121
    NUM_CHANELS = 1
    N_BLOCKS = 5
    MINI_BATCH = 28

    # Create the network
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(d1, d2, d3, NUM_CHANELS),
                                name='input'))
    for n in range(1, N_BLOCKS+1):
        model.add(layers.Conv3D(filters, kernel_size, STRIDES_CONV,
                                name='Block_{}_Conv1'.format(n)))
        model.add(layers.ReLU(name='Block_{}_Relu1'.format(n)))
        model.add(layers.Conv3D(filters, kernel_size, STRIDES_CONV,
                                name='Block_{}_Conv2'.format(n)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU(name='Block_{}_Relu2'.format(n)))
        model.add(layers.MaxPool3D(pool_size=(2, 2, 2),
                                   strides=STRIDES_MAXPOOL,
                                   name='Block_{}_MaxPool'.format(n)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model


if __name__ == '__main__':
    PROJECT_ROOT = Path('/regeage')
    data_path = PROJECT_ROOT / 'data' / 'BANC_2019'
    train_path = data_path / 'train_data'
    demographics_path = data_path / 'all_BANC_2019.csv'
    df = pd.read_csv(data_path / 'cleaned_BANC_2019.csv')
    rnd_seed = 1234

    # Get data and split into training and test
    data = get_data(train_path, df)
    test_size = .6
    X_train, X_test, y_train, y_test = train_test_split(data['imgs'],
                                                        data['labels'],
                                                        test_size=test_size,
                                                        random_state=rnd_seed)
    # Transform to correct format (samples x dim1 x dim2 x dim3 x channels)
    X_train = np.array(X_train)[:, :, :, :, np.newaxis]
    X_test = np.array(X_test)[:, :, :, :, np.newaxis]
    y_train = np.array(y_train, dtype='float32')
    y_test = np.array(y_test, dtype='float32')

    model = cnn_model()

    optimizer = optimizers.SGD(learning_rate=0.01,
                               momentum=0.9,
                               decay=0.0005)
    model.compile(optimizer=optimizer,
                  loss='mae')

    model.summary()

    model.fit(X_train, y_train,
              epochs=2,
              validation_data=(X_test, y_test))

    model.predict(X_test)

