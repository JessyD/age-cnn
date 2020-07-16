#!/usr/bin/env python3
"""
This code trains the model
"""
import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import (ReduceLROnPlateau, TensorBoard, ModelCheckpoint)

# Set random seeds
np.random.seed(1234)
tf.random.set_seed(1234)


def find_image_boundary(path):
    """ Find the limit of blank voxels in one image."""
    min_x = 1000
    max_x = 0
    min_y = 1000
    max_y = 0
    min_z = 1000
    max_z = 0

    img = nib.load(path).get_data()
    img = np.asarray(img, dtype='float32')
    img = np.nan_to_num(img)
    img_shape = img.shape

    #     X
    for i in range(0, img_shape[0]):
        if np.max(img[i, :, :]) > 0:
            break
    if min_x > i:
        min_x = i

    for i in range(img_shape[0] - 1, 0, -1):
        if np.max(img[i, :, :]) > 0:
            break
    if max_x < i:
        max_x = i

        #     Y
    for i in range(0, img_shape[1]):
        if np.max(img[:, i, :]) > 0:
            break
    if min_y > i:
        min_y = i

    for i in range(img_shape[1] - 1, 0, -1):
        if np.max(img[:, i, :]) > 0:
            break
    if max_y < i:
        max_y = i

        #     Z
    for i in range(0, img_shape[2]):
        if np.max(img[:, :, i]) > 0:
            break
    if min_z > i:
        min_z = i

    for i in range(img_shape[2] - 1, 0, -1):
        if np.max(img[:, :, i]) > 0:
            break
    if max_z < i:
        max_z = i

    return max_x, max_y, max_z, min_x, min_y, min_z, img


def load_img(img_path, min_x, max_x, min_y, max_y, min_z, max_z, mask, age):
    # Mask the image to have only brain related information
    img = nib.load(img_path).get_data()
    img = np.asarray(img, dtype='float32')
    img = np.nan_to_num(img)
    img = np.multiply(img, mask)
    img = img[min_x:max_x, min_y:max_y, min_z:max_z]
    # Create dummy dimension for channels
    img = img[..., np.newaxis]
    # Normalise all features
    img = np.true_divide(img, np.max(img))
    print("{:3}, {:3}, {:3}, {:3})\t{:}\t{:6.4} - {:6.4}".format(
        img.shape[0], img.shape[1], img.shape[2],
        img.shape[3], age, np.min(img), np.max(img)))
    return img


def load_data(train_path, ids_df, brain_mask):
    max_x, max_y, max_z, min_x, min_y, min_z, mask = find_image_boundary(brain_mask)
    print(max_x, max_y, max_z)
    print(min_x, min_y, min_z)

    data = {'subject_id': [], 'image': [], 'label': []}
    # Load dataset - 50 for the moment
    # TODO: Remove the limited number of subjects
    for index, row in df.iterrows():
        age = row['age']
        file_type = 'smwc1'
        base_path = train_path / row['Study'] / 'derivatives'
        spm_path = base_path / 'spm' / 'sub-{}'.format(row['Subject'])
        nifti = spm_path.glob(file_type + '*.nii')
        img_path = str(next(nifti))
        img = load_img(img_path, min_x, max_x, min_y, max_y,
                       min_z, max_z, mask, age)
        data['image'].append(img)
        data['subject_id'].append(row['Subject'])
        data['label'].append(row['age'])
        del img, age

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
    for n in range(1, N_BLOCKS + 1):
        model.add(layers.Conv3D(filters * filters_n[n - 1], kernel_size,
                                STRIDES_CONV, padding=conv_padding,
                                name='Block_{}_Conv1'.format(n)))
        model.add(layers.ReLU(name='Block_{}_Relu1'.format(n)))
        model.add(layers.Conv3D(filters * filters_n[n - 1], kernel_size,
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
    # Set random seeds
    rnd_seed = 42
    np.random.seed(rnd_seed)
    tf.random.set_seed(rnd_seed)

    # Alternative settings
    data_path = Path('/media/kcl_1/HDD/DATASETS/DOUTORADO_JESSICA/BANC_2019/')
    # PROJECT_ROOT = Path('/regeage')
    # data_path = PROJECT_ROOT / 'data' / 'BANC_2019'

    brain_mask = data_path / 'MNI152_T1_1.5mm_brain_masked2.nii.gz'
    train_path = data_path / 'train_data'
    demographics_path = data_path / 'all_BANC_2019.csv'

    # Output file structure
    model_path = data_path / 'cnn'
    model_path.mkdir(exist_ok=True, parents=True)
    log_dir = model_path / 'tensorboard'
    log_dir.mkdir(exist_ok=True, parents=True)
    weights_path = model_path / 'weights'
    weights_path.mkdir(exist_ok=True, parents=True)
    checkpoint_path = model_path / 'checkpoints'
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    # Get data and split into training and test
    ids_df = pd.read_csv(data_path / 'cleaned_BANC_2019.csv')
    # ids_df = ids_df[:100]  # for debug
    data = load_data(train_path, ids_df, brain_mask)

    test_size = .2
    idx_range = range(len(data['image']))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(data['image'],
                                                                             data['label'],
                                                                             idx_range,
                                                                             test_size=test_size,
                                                                             random_state=rnd_seed)

    print('Shape of training set {}'.format(X_train.shape))
    print('Shape of test set {}'.format(X_test.shape))

    # Dump the index of the train and test files
    train_df = pd.DataFrame(idx_train, columns=['idx_train'])
    train_df.to_csv(str(model_path / 'idx_train.csv'))

    test_df = pd.DataFrame(idx_test, columns=['idx_test'])
    test_df.to_csv(str(model_path / 'idx_test.csv'))

    # Create model
    model = cnn_model()
    optimizer = optimizers.SGD(learning_rate=0.005,
                               momentum=0.9,
                               decay=0.0005)

    model.compile(optimizer=optimizer, loss='mae')

    model.summary()

    # Callbacks
    # Checkpoint callback
    checkpoint_file = 'checkpoints-{epoch:02d}--{val_loss:.2f}.hdf'
    cp_callback = ModelCheckpoint(filepath=str(checkpoint_path / checkpoint_file),
                                  save_best_only=True,
                                  save_weights_only=True,
                                  monitor='val_loss',
                                  verbose=1)

    # Reduce learning rate when the metric has stopped improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=10,
                                  min_lr=0.00001)

    # Tensorboard callback
    experiment_log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard = TensorBoard(log_dir=log_dir/experiment_log_dir, profile_batch=0)

    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=200,
                        validation_data=(X_test, y_test),
                        callbacks=[reduce_lr, cp_callback, tboard])

    print('\nhistory dict:', history.history)

    model.predict(X_test,
                  batch_size=2,
                  verbose=1)

    # Serialise model to JSON format
    json_config = model.to_json()
    with open(str(model_path / 'model_config_banc2019.json'), 'w') as json_file:
        json_file.write(json_config)

    # Save weights of the trained model
    model.save_weights(str(weights_path / 'final_weights_banc2019'))

    # Write the entire model to HDF5
    model.save(str(model_path / 'banc2019.h5'), save_format='h5')