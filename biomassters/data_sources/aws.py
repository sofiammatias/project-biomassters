
import pandas as pd
from colorama import Fore, Style
import os
import awscli



    # First letter of each chip_id in 'features_metadata'
#    first_letter = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#                    'a', 'b', 'c', 'd', 'e', 'f']

#    for chunk_num, letter in enumerate(first_letter):

def get_aws_chunk(features: pd.DataFrame, raw_data_path:str,
                  features_path:str, agbm_s3_path:str, filter:str):
    """
    return a chunk of dataset in AWS, filtered by the dataframe 'features' and
    the filter string (chip_id characters to use)
    datafiles downloaded are tracked by 'features_metadata.csv'
    """
    # download data
    breakpoint()

    os.chdir(os.path.expanduser(raw_data_path))
    chunk_of_files = features[features.chip_id.str[:len(filter)] == filter]['s3path_eu']
    print (Fore.BLUE + f'\nDownloading files to {raw_data_path}...\n' + Style.RESET_ALL)
    path = os.path.dirname(chunk_of_files.iloc[0])
    end_name = os.path.basename(chunk_of_files.iloc[0]).split('_', 2)[2]
    aws_cli_agbm = f'aws s3 cp {agbm_s3_path} {raw_data_path} --recursive --exclude="*" --include="{filter}_agbm.tif" --no-sign-request'
    os.system(aws_cli_agbm)
    aws_cli_features = f'aws s3 cp {path} {raw_data_path} --recursive --exclude="*" --include="{filter}_S1_{end_name}" --include="{filter}_S2_{end_name}" --no-sign-request'
    os.system(aws_cli_features)

    # update 'features_metadata.csv' with downloaded files
    #original_features = pd.read_csv (os.getenv('FEATURES'))
    #datafiles = os.listdir(os.path.expanduser(raw_data_path))
    #datafiles_no_agbm = [item for item in datafiles if 'agbm' not in item]
    #original_features = original_features.loc[original_features['filename']
    #                                          .isin(datafiles_no_agbm),
    #                                          'file_downloaded'] = True
    #original_features.to_csv(os.path.expanduser(features_path), index = False)



    #if num_files_downloaded == len(features):
    #    print (Fore.BLUE + f'Finished downloading {num_files_downloaded} of {len(features)}'  + Style.RESET_ALL)
    #    features['file_downloaded'] = True
    #else:
    #    print (Fore.RED + f'Some files have not been downloaded: ({len(features) - num_files_downloaded})'  + Style.RESET_ALL)


def delete_aws_chunk():
    """
    delete a chunk of data from AWS after training model
    """
    pass

def features_per_month (features:pd.DataFrame, month:str) -> pd.DataFrame:
    """
    Filter 'features_metadata' for one month and return the filtered dataframe
    """
    breakpoint()
    return features[[features.month == month]]

def features_mode (features:pd.DataFrame, mode:str) -> pd.DataFrame:
    """
    Filter 'features_metadata' per using mode: train or test (uses 'split' column)
    """
    return features[[features.split == mode]]

def features_not_downloaded (features:pd.DataFrame) -> pd.DataFrame:
    """
    Filter 'features_metadata' per using mode: train or test (uses 'split' column)
    """
    return features[[features.file_downloaded == False]]
