"""
biomassters model package params
load and validate the environment variables in the `.env`
"""

import os
import numpy as np
import pandas as pd

# Important paths for project files and folders
LOCAL_DATA_PATH = os.path.expanduser(os.getenv('LOCAL_DATA_PATH'))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.getenv('LOCAL_REGISTRY_PATH'))

FEATURES_FILE = pd.read_csv (os.path.expanduser(os.getenv('FEATURES')))
TRAIN_AGBM_FILE = pd.read_csv (os.path.expanduser(os.getenv('TRAIN_AGBM')))
AGBM_S3_PATH = os.path.expanduser(os.getenv('TRAIN_AGBM_S3_PATH'))
FEATURES_TRAIN_S3_PATH = os.path.expanduser(os.getenv('FEATURES_TRAIN_S3_PATH'))
FEATURES_TEST_S3_PATH = os.path.expanduser(os.getenv('FEATURES_TEST_S3_PATH'))

# define usage mode: train or test
MODE = os.getenv('MODE')
# set month to download files
MONTH = os.getenv('MONTH')
# set the number of chip_id's for each 'chunk': 'chunk' size
CHIP_ID_SIZE = int(os.getenv('CHIP_ID_SIZE'))

# Parameters to setup S3 downaloading strings
filters= {'January': '04', 'February': '05', 'March': '06', 'April': '07',
          'May': '08', 'June': '09', 'July': '10', 'August': '11',
          'September': '00', 'October': '01', 'November': '02', 'December': '03',
          'All': '*', }

chip_id_letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'a', 'b', 'c', 'd', 'e', 'f']

combs = [val1+val2 for val1 in chip_id_letters for val2 in chip_id_letters]

# Others
chip_id_folder = 'Chip_Id'
PERC = float(os.getenv('PERC'))
