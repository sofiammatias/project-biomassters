
import pandas as pd
from colorama import Fore, Style
import os
import shutil
import tifffile

from biomassters.ml_logic.params import LOCAL_DATA_PATH, FEATURES_FILE, chip_id_folder

def features_per_month (features:pd.DataFrame, month:str) -> pd.DataFrame:
    """
    Filter 'features_metadata' for one month and return the filtered dataframe
    """
    return features[features.month == month]

def features_mode (features:pd.DataFrame, mode:str) -> pd.DataFrame:
    """
    Filter 'features_metadata' per using mode: train or test (uses 'split' column)
    """
    return features[features.split == mode.lower()]

def features_not_downloaded (features:pd.DataFrame) -> pd.DataFrame:
    """
    Filter 'features_metadata' per using mode: train or test (uses 'split' column)
    """
    return features[features.file_downloaded == False]

def features_downloaded (features:pd.DataFrame) -> pd.DataFrame:
    """
    Filter 'features_metadata' per using mode: train or test (uses 'split' column)
    """
    return features[features.file_downloaded == True]



def check_data_path (path):
    """
    Checks if path exists and creates if it doesn't
    """
    if not os.path.exists(os.path.expanduser(path)):
        os.makedirs(os.path.expanduser(path))
        print(Fore.BLUE + f'\nFolder {path} was created\n' + Style.RESET_ALL)


def organize_proj_folders (base_folder, old_path):
    """
    Create folders based on found dataset files.
    Assumes all data is only in LOCAL_DATA_PATH for automatic organization.
    - TrainData
      - ChipId
        - S1
        - S2
        - GroundTruth
    """
    files = os.listdir(old_path)
    files = [f for f in files if os.path.isfile(old_path+'/'+f)]
    features = FEATURES_FILE

    counter = 0
    for file in files:
        test_features = features[features.split == 'test']
        if True in test_features['filename'].str.contains(file).unique():
            first_folder = 'Test'
        else:
            first_folder = 'Train'
        new_path = f'{base_folder}{first_folder}/{chip_id_folder}/{file[:8]}'
        if 'S1' in file:
            new_path = f'{new_path}/S1'
            check_data_path(new_path)
            shutil.copy (f'{old_path}/{file}', f'{new_path}/{file}')
        elif 'S2' in file:
            new_path = f'{new_path}/S2'
            check_data_path(new_path)
            shutil.copy (f'{old_path}/{file}', f'{new_path}/{file}')
        elif 'agbm' in file:
            new_path = f'{new_path}/GroundTruth'
            check_data_path(new_path)
            shutil.copy (f'{old_path}/{file}', f'{new_path}/{file}')
        else:
            print (f"The path {old_path} doesn't have any dataset files")
            counter += 1
            if counter == 10:
                return None




def organize_folders ():
    """
    Organizes downloaded data into folders assuming data is in LOCAL_DATA_PATH
    """
    base_folder = LOCAL_DATA_PATH
    check_data_path (base_folder)
    old_path = base_folder
    organize_proj_folders (base_folder, old_path)



def organize_folders_user ():
    """
    Organizes downloaded data into folders with a folder path inserted by user.
    """
    base_folder = LOCAL_DATA_PATH
    check_data_path (base_folder)

    old_path = input ('Enter dataset full path:')
    if old_path == '':
        old_path = base_folder
    organize_proj_folders (base_folder, old_path)



def image_to_np(path, filename):
    """
    Read tif file and get a numpy array
    """
    file = os.path.join(f'{path}{filename}')
    return tifffile.imread(file)
