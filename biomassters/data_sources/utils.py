
import pandas as pd
from colorama import Fore, Style
import os
import shutil
import tifffile

from biomassters.ml_logic.params import LOCAL_DATA_PATH

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

def check_data_path (path):
    """
    Checks if path exists and creates if it doesn't
    """
    if not os.path.exists(os.path.expanduser(path)):
        os.makedirs(os.path.expanduser(path))
        print(Fore.BLUE + f'\nFolder {path} was created\n' + Style.RESET_ALL)



def organize_folders ():
    """
    Organize downloaded data into folders.
    Assumes all data is only in LOCAL_DATA_PATH for automatic organization.
    - TrainData
      - ChipId
        - S1
        - S2
        - GroundTruth
    """
    base_folder = LOCAL_DATA_PATH
    check_data_path (base_folder)
    old_path = base_folder
    files = os.listdir(old_path)
    files = [f for f in files if os.path.isfile(old_path+'/'+f)]
    for file in files:
        if 'S1' in file:
            new_path = f'{base_folder}Train/Chip_Id/{file[:8]}/S1'
            check_data_path(new_path)
            shutil.copy (f'{old_path}/{file}', f'{new_path}/{file}')
        elif 'S2' in file:
            new_path = f'{base_folder}Train/Chip_Id/{file[:8]}/S2'
            check_data_path(new_path)
            shutil.copy (f'{old_path}/{file}', f'{new_path}/{file}')
        elif 'agbm' in file:
            new_path = f'{base_folder}Train/Chip_Id/{file[:8]}/GroundTruth'
            check_data_path(new_path)
            shutil.copy (f'{old_path}/{file}', f'{new_path}/{file}')
        else:
            print (f"The path {old_path} doesn't have any dataset files")
            return None



def organize_folders_user ():
    """
    Organizes downloaded data into folders with a folder inserted by user.
    - TrainData
      - ChipId
        - S1
        - S2
        - GroundTruth
    """
    base_folder = LOCAL_DATA_PATH
    check_data_path (base_folder)

    old_path = input ('Enter dataset full path:')
    if old_path == '':
        old_path = base_folder
    files = os.listdir(old_path)
    files = [f for f in files if os.path.isfile(old_path+'/'+f)]
    for file in files:
        if 'S1' in file:
            new_path = f'{base_folder}Train/Chip_Id/{file[:8]}/S1'
            check_data_path(new_path)
            shutil.copy (f'{old_path}/{file}', f'{new_path}/{file}')
        elif 'S2' in file:
            new_path = f'{base_folder}Train/Chip_Id/{file[:8]}/S2'
            check_data_path(new_path)
            shutil.copy (f'{old_path}/{file}', f'{new_path}/{file}')
        elif 'agbm' in file:
            new_path = f'{base_folder}Train/Chip_Id/{file[:8]}/GroundTruth'
            check_data_path(new_path)
            shutil.copy (f'{old_path}/{file}', f'{new_path}/{file}')
        else:
            print (f"The path {old_path} doesn't have any dataset files")
            return None



def image_to_np(path, filename):
    """
    Read tif file and get a numpy array
    """
    file = os.path.join(f'{path}{filename}')
    return tifffile.imread(file)
