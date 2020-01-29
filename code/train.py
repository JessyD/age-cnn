#!/usr/bin/env python3
"""
This code trains the model
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import nibabel as nib
from nilearn.masking import apply_mask
from sklearn.model_selection import train_test_split


def load_data_npz(train_path, df):
    # Load dataset - 50 for the moment
    data = {'subject_id': [], 'image': [], 'label': []}
    # TODO: Remove the limited number of subjects
    for index, row in df.iloc[:50].iterrows():
        path = train_path / row['Study'] / 'derivatives' / 'npz' / str('sub-' +
                                                                      row['Subject'])
        nifti = path.glob('smwc1*.npz')
        img = str(next(nifti))
        # Mask the image to have only brain related information
        img = np.load(img)
        data['image'].append(img['image'])
        data['subject_id'].append(row['Subject'])
        data['label'].append(img['label'])

    # Transform to correct format (samples x dim1 x dim2 x dim3 x channels)
    data['image'] = np.array(data['image'])
    data['label'] = np.array(data['label'])
    data['subject_id'] = np.array(data['subject_id'])
    return data


def cnn_model():
    # Define parameters for the network
    filters = 8
    kernel_size = 3
    STRIDES_CONV = 1
    STRIDES_MAXPOOL = 2
    conv_padding = 'same'
    pool_padding = 'valid'
    # d1 = 121
    # d2 = 145
    # d3 = 121
    d1 = 99
    d2 = 123
    d3 = 104
    NUM_CHANELS = 1
    N_BLOCKS = 5
    filters_n = [1, 2, 4, 8, 16]

    # Create the network
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(d1, d2, d3, NUM_CHANELS),
                                name='input'))
    for n in range(1, N_BLOCKS+1):
        model.add(layers.Conv3D(filters * filters_n[n-1], kernel_size,
                                STRIDES_CONV, padding=conv_padding,
                                name='Block_{}_Conv1'.format(n)))
        model.add(layers.ReLU(name='Block_{}_Relu1'.format(n)))
        model.add(layers.Conv3D(filters * filters_n[n-1], kernel_size,
                                STRIDES_CONV, padding=conv_padding,
                                name='Block_{}_Conv2'.format(n)))
        if not n == 5:
            model.add(layers.BatchNormalization())
        model.add(layers.ReLU(name='Block_{}_Relu2'.format(n)))
        model.add(layers.MaxPool3D(pool_size=(2, 2, 2),
                               strides=STRIDES_MAXPOOL,
                               padding=pool_padding,
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
    MINI_BATCH = 28

    # Get data and split into training and test
    data = load_data_npz(train_path, df)
    test_size = .6
    X_train, X_test, y_train, y_test = train_test_split(data['image'],
                                                        data['label'],
                                                        test_size=test_size,
                                                        random_state=rnd_seed)
    model = cnn_model()

    optimizer = optimizers.SGD(learning_rate=0.0001,
                               momentum=0.9,
                               decay=0.0005)
    model.compile(optimizer=optimizer,
                  loss='mae')

    model.summary()

    model.fit(X_train, y_train, batch_size=6,
              epochs=2,
              validation_data=(X_test, y_test))

    model.predict(X_test, verbose=1, batch_size=2)

