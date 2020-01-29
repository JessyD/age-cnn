from pathlib import Path
import re
import glob

import pandas as pd
import nibabel as nib
import numpy as np


class PreprocessData(object):
    def __init__(self, data_path, shuffle=True, batch_size=None):
        self.data_path = data_path
        self.shuffle = shuffle
        self.batchsize = batch_size

    def find_image_boundary(path):
        """ Find the limit of blank voxels in one image.

        :param path:
        :return:
        """
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

    def transform_data_npz(train_path, df, mask):
        max_x, max_y, max_z, min_x, min_y, min_z, mask = PreprocessData.find_image_boundary(brain_mask)
        print(max_x, max_y, max_z)
        print(min_x, min_y, min_z)

        # Load dataset - 50 for the moment
        # TODO: Remove the limited number of subjects
        for index, row in df.iloc[:50].iterrows():
            file_type = 'smwc1'
            base_path = train_path / row['Study'] /'derivatives'
            spm_path = base_path / 'spm' / str('sub-' + index)
            nifti = spm_path.glob(file_type + '*.nii')
            img = str(next(nifti))
            # Mask the image to have only brain related information
            img = nib.load(img).get_data()
            img = np.asarray(img, dtype='float32')
            img = np.nan_to_num(img)
            img = np.multiply(img, mask)
            img = img[min_x:max_x, min_y:max_y, min_z:max_z]
            # Create dummy dimension for channels
            img = img[..., np.newaxis]
            # Normalise all features
            img = np.true_divide(img, np.max(img))
            print("{:3}, {:3}, {:3}, {:3})\t{:}\t{:6.4} - {:6.4}".format(img.shape[0],
                img.shape[1], img.shape[2], img.shape[3], row['Age'], np.min(img), np.max(img)))
            npz_path = base_path / 'npz' / str('sub-' + index)
            npz_path.mkdir(exist_ok=True, parents=True)
            file_name = npz_path / '{}_sub-{}.npz'.format(file_type, index)
            np.savez(file_name, image=img, label=row['Age'])
            del img

    def clean_ixi_dataset(ixi_path):
        """
        Remove the hospital name from the files folder
        """
        print('Cleaning IXI dataset')
        for folder in ixi_path.glob('sub-*'):
            print(folder)
            new_folder = Path(str(folder).replace(folder.stem, folder.stem[:10]))
            folder.rename(new_folder)

    def clean_nki_dataset(nki_path):
        """
        Remove RK from the folder name
        """
        new_path = 'NKI'
        nki_path.rename(Path(nki_path.parent, new_path))

    def clean_sald_dataset(sald_path):
        """
        Remove the additional 0s from the file name
        """
        for subject_folder in sald_path.glob('sub-*'):
            subject_id = subject_folder.name[:4] + subject_folder.name[5:]
            new_subject_folder = Path(str(subject_folder).replace(subject_folder.stem, subject_id))
            subject_folder.rename(new_subject_folder)

    def load_labels(demographics_path):
        df = pd.read_csv(demographics_path)
        # Remove unnecessary columns
        df.drop(columns=['File.location', 'Site'], inplace=True)
        # Clean the subject's ID for the GSP dataset
        idx = df[df['Study'] == 'GSP'].index
        df.loc[idx, ('Subject')] = df.loc[idx, ('Subject')].apply(lambda x: re.sub('_S1', '', x))

        # Duplicates
        print(df.duplicated())
        # TODO: for now drop the duplicates. We will keep the first occurrence of
        # each duplicate
        df.drop_duplicates(subset='Subject', inplace=True)
        df.set_index('Subject', inplace=True)
        return df

    def find_subjects(df, data_path):
        """
        Clean demographics so that we have only subjects for which we have data
        """
        print('Demographic shape: {}'.format(df.shape))
        for study in df['Study'].unique():
            study_folder = data_path / study / 'derivatives' / 'spm'
            for subject in df[df['Study'] == study].index:
                subject_folder = study_folder / ('sub-' + subject)
                # Delete this entrance from the dataframe if we don't have the
                # image for that subject
                if not subject_folder.is_dir():
                    df.drop(subject, inplace=True)
                    print(subject_folder)
        print('Demographic shape: {}'.format(df.shape))
        return df

    def save_nifti_images_tfrecords():
        """
        Load all nifti images and save them as tfrecords
        """


if __name__ == '__main__':
    PROJECT_ROOT = Path('/regeage')
    data_path = PROJECT_ROOT / 'data' / 'BANC_2019'
    train_path = data_path / 'train_data'
    demographics_path = data_path / 'all_BANC_2019.csv'

    # Clean the IXI, the NKI, SALD  path to make sure that the paths and
    # subjects'id between demographics files and stored files are consistent
    clean_ixi = False
    if clean_ixi:
        ixi_path = train_path / 'IXI' / 'derivatives' / 'spm'
        PreprocessData.clean_ixi_dataset(ixi_path)
    clean_nki = False
    if clean_nki:
        nki_path = train_path / 'NKI_RS'
        PreprocessData.clean_nki_dataset(nki_path)
    clean_sald = False
    if clean_sald:
        sald_path = train_path / 'SALD' / 'derivatives' / 'spm'
        PreprocessData.clean_sald_dataset(sald_path)

    dataset = PreprocessData(data_path, shuffle=True)
    df = PreprocessData.load_labels(demographics_path)
    df = PreprocessData.find_subjects(df, train_path)
    # Save the cleaned dataframe
    df.to_csv(data_path / 'cleaned_BANC_2019.csv')

    # Load the mask image
    brain_mask = data_path / 'MNI152_T1_1.5mm_brain_masked2.nii.gz'
    PreprocessData.transform_data_npz(train_path, df, brain_mask)
