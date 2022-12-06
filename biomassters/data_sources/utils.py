
import pandas as pd
import numpy as np
from colorama import Fore, Style
import os
import shutil
from biomassters.ml_logic.params import LOCAL_DATA_PATH, FEATURES_FILE, chip_id_folder
from biomassters.ml_logic.params import FEATURES_FILE_PATH, chip_id_folder


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
    print(Fore.BLUE + f'\n✅ All dataset files  from {old_path} organized\n' + Style.RESET_ALL)




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
    print(Fore.BLUE + f'\n✅ All dataset files from {old_path} organized\n' + Style.RESET_ALL)



def check_for_downloaded_files():
    """
    Checks for dataset downloaded files in LOCAL_DATA_PATH and updates 'features_
    metadata' column 'file_downloaded' accordingly to keep track of files already
    downloaded
    """
    # Add checksum here if there's time

    # Updates 'features_metadata.csv' with newly downloaded data
    datafiles = [file for _,_,f in os.walk(LOCAL_DATA_PATH) for file in f]
    datafiles_no_agbm = [item for item in datafiles if 'agbm' not in item]
    features = FEATURES_FILE
    if 'file_downloaded' not in features.columns:
        features['file_downloaded'] = False
    features['file_downloaded'] = features['filename'].isin(pd.Series(datafiles_no_agbm))
    features.to_csv(os.path.expanduser(FEATURES_FILE_PATH), index = False)
    print (Fore.GREEN + f"\n✅ 'features_metadata.csv' updated with downloaded files\n" + Style.RESET_ALL)


def set_trained_files(chip_id_list:np.ndarray):
    """
    Once a model is saved, the list of chip_ids is updated in 'features_metadata'
    in a new column 'file_trained' to keep track of files already trained in the model
    """
    # Updates 'features_metadata.csv' with newly trained data
    features = FEATURES_FILE
    if 'file_trained' not in features.columns:
        features['file_trained'] = False
    features['file_trained'] = features['chip_id'].isin(pd.Series(chip_id_list))
    features.to_csv(os.path.expanduser(FEATURES_FILE_PATH), index = False)
    print (Fore.GREEN + f"\n✅ 'features_metadata.csv' updated with trained files\n" + Style.RESET_ALL)


def set_predicted_chip_ids(chip_id_list:np.ndarray):
    """
    Once a chunk of predictions is done, the list of chip_ids is updated in 'features_metadata'
    in a new column 'chip_id_predict' to keep track of chip ids already predicted in the model
    """
    # Updates 'features_metadata.csv' with newly trained data
    features = FEATURES_FILE
    if 'file_predict' not in features.columns:
        features['chip_id_predict'] = False
    features['chip_id_predict'] = features['chip_id'].isin(pd.Series(chip_id_list))
    features.to_csv(os.path.expanduser(FEATURES_FILE_PATH), index = False)
    print (Fore.GREEN + f"\n✅ 'features_metadata.csv' updated with trained files\n" + Style.RESET_ALL)
