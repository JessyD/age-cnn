"""
Load Tensorflow model and make inferences
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from train import find_image_boundary, load_img, cnn_model


def load_test_data(experiment_path, ids_df, brain_mask):
    max_x, max_y, max_z, min_x, min_y, min_z, mask = find_image_boundary(brain_mask)
    print(max_x, max_y, max_z)
    print(min_x, min_y, min_z)

    data = {'subject_id': [], 'image': [], 'label': [], 'sescode': []}
    for index, row in ids_df.iterrows():
        age = row['age']
        file_type = 'smwc1'
        base_path = (experiment_path / row['participant_id'] /
                     row['sescode'] / 'anat' / 'brainager_v21')
        # if we do not have imaging info for that line go to next
        if not base_path.is_dir():
            continue
        nifti = base_path.glob(file_type + '*.nii')
        img_path = str(next(nifti))
        img = load_img(img_path, min_x, max_x, min_y, max_y,
                       min_z, max_z, mask, age)
        data['image'].append(img)
        data['subject_id'].append(row['participant_id'])
        data['label'].append(row['age'])
        data['sescode'].append(row['sescode'])
        del img, age

    # Transform to correct format (samples x dim1 x dim2 x dim3 x channels)
    data['image'] = np.array(data['image'])
    data['label'] = np.array(data['label'])
    data['subject_id'] = np.array(data['subject_id'])
    return data


if __name__ == '__main__':
    PROJECT_ROOT = Path('/regeage')
    # cnn_path = PROJECT_ROOT / 'data' / 'BANC_2019'/ 'cnn'
    data_path = PROJECT_ROOT / 'data'
    cnn_path = data_path / 'cnn'

    brain_mask = data_path / 'MNI152_T1_1.5mm_brain_masked2.nii.gz'

    # load model
    checkpoint_path = '.outputs/cnn/checkpoints/checkpoints-94--4.37.hdf'
    model = cnn_model()
    model.load_weights(checkpoint_path)

    # print model summary
    model.summary()

    # Load the new datasets to evaluate from
    experiment = 'myConnectome'
    myconnectome_path = PROJECT_ROOT / 'data' / '{}_cleaned'.format(experiment)
    # Demographics for simon
    myconnectome_df = pd.read_csv(str(myconnectome_path / 'participants.tsv'),
                                 sep='\t', header=0)
    # Drop nan for age
    # TODO: Add this the drop step a preprocessing step
    myconnectome_df.dropna(inplace=True)
    data = load_test_data(myconnectome_path, myconnectome_df, brain_mask)

    # Check main performance on the dataset
    score = model.evaluate(data['image'], data['label'])

    # Predict age
    predicted_age = np.squeeze(model.predict(data['image']))
    dic = {'subject_id': data['subject_id'], 'sescode': data['sescode'],
           'predicted age': predicted_age}
    predicted_df_myconnectome = pd.DataFrame(dic)
    predicted_df_myconnectome.to_csv(str(myconnectome_path /
                                     'cnn_cole_myconnectome_predicted.csv'),
                                     index=False)
