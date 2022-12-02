from biomassters.ml_logic.params import LOCAL_DATA_PATH, MODE, MONTH
from biomassters.ml_logic.params import chip_id_folder, CHIP_ID_SIZE, PERC
from biomassters.ml_logic.model import image_to_np

#from taxifare.data_sources.local_disk import (get_pandas_chunk, save_local_chunk)
#from taxifare.data_sources.big_query import (get_bq_chunk, save_bq_chunk)

import os
import pandas as pd
import numpy as np
from biomassters.data_sources.utils import image_to_np


def import_data():
    """
    import images data and transfors it into 2 arrays of 4 dimensions:
    X1 (X_files, 256, 256, channels_num = 4)
    X1 (X_files, 256, 256, channels_num = 11)
    besides transforming the tif images into numpy arrays, a scaling is done with
    tensorflow.image.per_image_standardization
    this will consist on the main dataset
    it also imports the target agbm image (y)
    """

    # get Xs and ys
    X1 = []
    X2 = []
    y = []
    n_chips = CHIP_ID_SIZE
    basepath = os.path.expanduser(f'{LOCAL_DATA_PATH}{MODE.capitalize()}/{chip_id_folder}')
    chip_ids = os.listdir(basepath)


    #chip_ids = list_chip_ids[: int(len(list_chip_ids) * PERC)]
    #filt_features = features_per_month(FEATURES_FILE, MONTH)
    #filt_features = features_mode(filt_features, MODE)
    #filt_features = features_downloaded(filt_features)
    #filt_features = filt_features[filt_features['chip_id'] == chip_ids]
    #basepath = '../raw_data/Train/Chip_Id/'

    for x in range(0, n_chips):
        path = os.path.join(basepath, chip_ids[x])
        path1_1 = os.path.join(path, 'S1')
        path1_2 = os.path.join(path, 'S2')
        path1_3 = os.path.join(path, 'GroundTruth')
        files_list = [file for file in os.listdir(path1_1)]
        breakpoint()
        files_list.sort()
        files_f = files_list[-5:]
        for x in range(0,len(files_f)):
            path2_1 = os.path.join(path1_1, files_f[x])
            X1.append(image_to_np(path2_1))
        files_list = [file for file in os.listdir(path1_2)]
        files_list.sort()
        files_f = files_list[-5:]
        for x in range(0,len(files_f)):
            path2_2 = os.path.join(path1_2, files_f[x])
            X2.append(image_to_np(path2_2))
        y.append (np.asarray(os.listdir(path1_3)))

    return np.asarray(X1), np.asarray(X2), np.asarray(y), chip_ids

def save_predictions(y_pred):
    pass
    # Build a function that saves one numpy array into a tif image and stores it in LOCAL_OUTPUT_DATA


#def get_chunk(source_name: str,
#              index: int = 0,
#              chunk_size: int = None,
#              verbose=False) -> pd.DataFrame:
#    """
#    Return a `chunk_size` rows from the source dataset, starting at row `index` (included)
#    Always assumes `source_name` (CSV or Big Query table) have headers,
#    and do not consider them as part of the data `index` count.
#    """
#
#    if "processed" in source_name:
#        columns = None
#        dtypes = DTYPES_PROCESSED_OPTIMIZED
#    else:
#        columns = COLUMN_NAMES_RAW
#        if os.environ.get("DATA_SOURCE") == "big query":
#            dtypes = DTYPES_RAW_OPTIMIZED
#        else:
#            dtypes = DTYPES_RAW_OPTIMIZED_HEADLESS
#
#    if os.environ.get("DATA_SOURCE") == "big query":
#
#
#        chunk_df = get_bq_chunk(table=source_name,
#                                index=index,
#                                chunk_size=chunk_size,
#                                dtypes=dtypes,
#                                verbose=verbose)
#
#        return chunk_df
#
#    chunk_df = get_pandas_chunk(path=source_name,
#                                index=index,
#                                chunk_size=chunk_size,
#                                dtypes=dtypes,
#                                columns=columns,
#                                verbose=verbose)
#
#    return chunk_df


#def save_chunk(destination_name: str,
#               is_first: bool,
#               data: pd.DataFrame) -> None:
#    """
#    save chunk
#    """
#
#    if os.environ.get("DATA_SOURCE") == "big query":
#
#
#         save_bq_chunk(table=destination_name,
#                      data=data,
#                      is_first=is_first)
#
#        return
#
#    save_local_chunk(path=destination_name,
#                     data=data,
#                     is_first=is_first)
