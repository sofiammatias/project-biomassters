"""
taxifare model package params
load and validate the environment variables in the `.env`
"""

import os
import numpy as np
import pandas as pd

LOCAL_DATA_PATH = os.path.expanduser(os.getenv('LOCAL_DATA_PATH'))
FEATURES_PATH = os.path.expanduser(os.getenv('FEATURES'))
AGBM_S3_PATH = os.path.expanduser(os.getenv('TRAIN_AGBM_S3_PATH'))
FEATURES_TRAIN_S3_PATH = os.path.expanduser(os.getenv('FEATURES_TRAIN_S3_PATH'))

# 'features_metadata'
FEATURES = pd.read_csv(os.path.expanduser(os.getenv('FEATURES')))
# define usage mode: train or test
MODE = os.getenv('MODE')
# set month to download files
MONTH = os.getenv('MONTH')
# set the number of chip_id's for each 'chunk': 'chunk' size
CHIP_ID_SIZE = int(os.getenv('CHIP_ID_SIZE'))
