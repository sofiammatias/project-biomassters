from biomassters.ml_logic.params import LOCAL_DATA_PATH, LOCAL_OUTPUT_PATH, MODE
from biomassters.ml_logic.params import chip_id_folder, CHIP_ID_SIZE, MONTH
from biomassters.ml_logic.params import FEATURES_FILE, CHIP_ID_PATH
from biomassters.ml_logic.model import image_standard, get_X1, get_X2, get_y
from biomassters.data_sources.utils import check_data_path

import time

#from taxifare.data_sources.local_disk import (get_pandas_chunk, save_local_chunk)
#from taxifare.data_sources.big_query import (get_bq_chunk, save_bq_chunk)

import os, tifffile, datetime
from colorama import Fore, Style
import pandas as pd
import numpy as np


def create_chip_ids_list():
    basepath = os.path.expanduser(f'{LOCAL_DATA_PATH}{MODE.capitalize()}/{chip_id_folder}')
    all_chip_ids = os.listdir(basepath)
    all_chip_ids.sort()
    chip_ids = []
    for chip_id in all_chip_ids:
        if (os.path.exists(f'{basepath}/{chip_id}/S1')
            and os.path.exists(f'{basepath}/{chip_id}/S2')
            and os.path.exists(f'{basepath}/{chip_id}/GroundTruth')):
            chip_ids.append (chip_id)
    return chip_ids



def import_data(i, chip_ids):
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
    X1=get_X1(chip_ids[i])
    X2=get_X2(chip_ids[i])
    y=get_y(chip_ids[i])

    y=np.asarray(y)
    X1=np.mean(np.asarray(X1),axis=0)
    X1=image_standard(X1)
    X2=np.mean(np.asarray(X2),axis=0)
    X2=image_standard(X2)

    return X1, X2, y


def get_predictions(model):
    basepath = os.path.expanduser(f'{LOCAL_DATA_PATH}{MODE.capitalize()}/{chip_id_folder}')
    predicts=[] # file with predictions: necessary for API
    i = 0

    chip_ids = create_chip_ids_list()

    while i <= len(chip_ids):
        X1 = []
        X2 = []
        path = os.path.join(basepath, chip_ids[i])
        path1_1 = os.path.join(path, 'S1')
        path1_2 = os.path.join(path, 'S2')
        files_list = [file for file in os.listdir(path1_1)]
        for x in range(0,len(files_list)):
            path2_1 = os.path.join(path1_1, files_list[x])
            img = tifffile.imread(path2_1)
            X1.append(img)
        files_list = [file for file in os.listdir(path1_2)]
        for x in range(0,len(files_list)):
            path2_2 = os.path.join(path1_2, files_list[x])
            img = tifffile.imread(path2_2)
            X2.append(img)
        X1=np.mean(np.asarray(X1),axis=0)
        X1=image_standard(X1)
        X2=np.mean(np.asarray(X2),axis=0)
        X2=image_standard(X2)
        print(f"Predicting chip {chip_ids[i]}.")
        prediction = model.predict([X1, X2])
        predict = np.asarray(prediction)
        predicts.append (predict)
        path=f'{LOCAL_OUTPUT_PATH}/'
        check_data_path(path)
        tifffile.imwrite(f'{path}{chip_ids[i]}_agbm.tif', predict.reshape((256,256)))
        print(f"Done with chip {chip_ids[i]}!")
        i += 1
        if i == len(chip_ids):
            print(f"\n✅ Prediction of {i} chip done: images with shape ", predict.shape)
            break
    return predicts, chip_ids
