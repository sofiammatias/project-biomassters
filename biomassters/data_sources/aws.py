
import pandas as pd
import numpy as np
from colorama import Fore, Style
import os
import awscli


def get_aws_chunk(features: pd.DataFrame, raw_data_path:str,
                  agbm_s3_path:str, chip_id:np.ndarray, num_file:str):
    """
    return a chunk of dataset in AWS, filtered by the dataframe 'features' and
    chip_id + num_file strings (chip_id characters to use and the numbering at the end)
    """

    os.chdir(os.path.expanduser(raw_data_path))
    print (Fore.BLUE + f'\nDownloading files to {raw_data_path}...\n' + Style.RESET_ALL)
    features_path = os.path.dirname(features['s3path_eu'].iloc[0])
    string_agbm = ''
    stringS1_features = ''
    stringS2_features = ''
    if type(chip_id) == np.ndarray and len(chip_id) > 1:
        for chip in chip_id:
            string_agbm += f' --include="{chip}_agbm.tif" '
            stringS1_features += f' --include="{chip}_S1_{num_file}.tif" '
            stringS2_features += f' --include="{chip}_S2_{num_file}.tif" '
    elif type(chip_id) == np.ndarray and len(chip_id) == 1:
        string_agbm = f' --include="{chip_id[0]}_agbm.tif" '
        stringS1_features = f' --include="{chip_id[0]}_S1_{num_file}.tif" '
        stringS2_features = f' --include="{chip_id[0]}_S2_{num_file}.tif" '
    else:
        print (Fore.RED + f'\nError in downloading files. Aborting... \n' + Style.RESET_ALL)
        return None

    aws_cli_agbm = f'aws s3 cp {agbm_s3_path} {raw_data_path} --recursive --exclude="*" {string_agbm} --no-sign-request'
    os.system(aws_cli_agbm)
    aws_cli_features = f'aws s3 cp {features_path} {raw_data_path} --recursive --exclude="*" {stringS1_features} {stringS2_features} --no-sign-request'
    os.system(aws_cli_features)
    print (Fore.GREEN + f'\n✅ Finished downloading files to {raw_data_path}.\n' + Style.RESET_ALL)
    return None


def delete_aws_chunk():
    """
    delete a chunk of data from AWS after training model
    """
    pass
