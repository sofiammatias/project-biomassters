
from colorama import Fore, Style

import time
print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)
start = time.perf_counter()
import tensorflow
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras import Input, Model
from tensorflow import image, expand_dims
from keras.callbacks import EarlyStopping
from keras.metrics import RootMeanSquaredError as rmse
end = time.perf_counter()
print(f"\n✅ tensorflow loaded ({round(end - start, 2)} secs)")

from typing import Tuple
import os
import tifffile
import numpy as np
from biomassters.ml_logic.params import CHIP_ID_PATH
#from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler


def get_X1(chip_id):
    basepath = CHIP_ID_PATH
    X1 = []
    path = os.path.join(basepath, chip_id)
    path1_1 = os.path.join(path, 'S1')
    files_list = [file for file in os.listdir(path1_1)]
    for x in range(0,len(files_list)):
        path2_1 = os.path.join(path1_1, files_list[x])
        img = tifffile.imread(path2_1)
        X1.append(img)
    return X1

def get_X2(chip_id):
    basepath = CHIP_ID_PATH
    X2=[]
    path = os.path.join(basepath, chip_id)
    path1_2 = os.path.join(path, 'S2')
    files_list = [file for file in os.listdir(path1_2)]
    for x in range(0,len(files_list)):
        path2_2 = os.path.join(path1_2, files_list[x])
        img = tifffile.imread(path2_2)
        X2.append(img)
    return X2


def get_y(chip_id):
    y=[]
    basepath = CHIP_ID_PATH
    path = os.path.join(basepath, chip_id)
    path1_3 = os.path.join(path, 'GroundTruth')
    for file in os.listdir(path1_3):
        path2_3 = os.path.join(path1_3, file)
        img = tifffile.imread(path2_3)
        img = tensorflow.expand_dims(img,axis=0)
        y.append(img)
    return y


def image_standard(X1):
    """
    Read tif file and get a numpy array with scaling applied and a dimensions added for
    modelling purposes (dimensions matching)
    """
    X1 = image.per_image_standardization(X1)
    X1 = expand_dims(X1, axis=0)

    return X1


def image_to_np(path):
    """
    Read tif file and get a numpy array with scaling applied and a dimensions added for
    modelling purposes (dimensions matching)
    """
    img = tifffile.imread(path)
    img = image.per_image_standardization(img)
    img = expand_dims(img, axis=0)

    return img


def image_to_np_y(path):
    """
    Read tif file and get a numpy array with scaling applied and a dimensions added for
    modelling purposes (dimensions matching)
    """
    img = tifffile.imread(path)
    img = expand_dims(img, axis=0)

    return img



def initialize_model(start_neurons = 32) -> Model:
    """
    Initialize the Neural Network for image processing
    """
    print("Initialize model..." )
    input1 = Input(shape=(256,256,4))
    input2 = Input(shape=(256,256,11))

    conv1 = Conv2D(start_neurons * 1, (4, 4), activation="relu", padding="same")(input1)
    conv1_1 = Conv2D(start_neurons * 1, (4, 4), activation="relu", padding="same")(input2)
    conv1 = concatenate([conv1, conv1_1])
    conv1 = Conv2D(start_neurons * 1, (4, 4), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (4, 4), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (4, 4), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)


    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = Conv2D(1, (1,1), padding="same", activation="linear")(uconv1)

    model = Model(inputs=[input1, input2], outputs = [output_layer])

    print("\n✅ model initialized")

    return model



def compile_model(model: Model) -> Model:
    """
    Compile the Neural Network
    """
    model.compile(loss='mse',
                  optimizer="rmsprop",
                  metrics=rmse())
    print("\n✅ model compiled and fitted")
    return model


def train_model(model: Model,
                X1: np.ndarray,
                X2: np.ndarray,
                y: np.ndarray,) -> Tuple[Model, dict]:

    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)
    es = EarlyStopping(restore_best_weights = True)
    history = model.fit([X1,X2], y, verbose = 1, epochs = 10, callbacks = [es])

    return model, history


def evaluate_model(model: Model,
                   X1: np.ndarray,
                   X2: np.ndarray,
                   y: np.ndarray) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X1)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(x=[X1, X2],
                                 y=y,
                                 verbose=1,
                                 return_dict=True)

    loss = metrics["mse"]
    rmse = metrics["mse"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} rmse {round(rmse, 2)}")

    return metrics
