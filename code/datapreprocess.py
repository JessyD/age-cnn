from pathlib import Path
import re
import glob

import pandas as pd
import nibabel as nib


class PreprocessData(object):
    def __init__(self, data_path, shuffle=True, batch_size=None):
        self.data_path = data_path
        self.shuffle = shuffle
        self.batchsize = batch_size

    def read_mri_data(path):
        """
        Function to load the nifti file and transform it to tensorflow object
        """

        # image size
        width = 121
        height = 145
        depth = 121

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




