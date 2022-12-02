import numpy as np
import pandas as pd
import os
import tifffile

from colorama import Fore, Style

from biomassters.data_sources.aws import get_aws_chunk
from biomassters.data_sources.utils import features_not_downloaded
from biomassters.data_sources.utils import features_downloaded, check_data_path
from biomassters.data_sources.utils import features_per_month, features_mode
from biomassters.ml_logic.params import LOCAL_DATA_PATH, FEATURES_FILE, MODE, MONTH
from biomassters.ml_logic.params import AGBM_S3_PATH ,FEATURES_TRAIN_S3_PATH
from biomassters.ml_logic.params import FEATURES_TEST_S3_PATH, CHIP_ID_SIZE, PERC
from biomassters.ml_logic.params import filters, chip_id_letters, combs, chip_id_folder
from biomassters.ml_logic.data import get_chunk


from biomassters.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from biomassters.ml_logic.preprocessor import preprocess_features
from biomassters.ml_logic.registry import get_model_version

from biomassters.ml_logic.registry import load_model, save_model

def load_all_dataset():
    raw_data_path = LOCAL_DATA_PATH
    agbm_s3_path = AGBM_S3_PATH
    features_train_path = FEATURES_TRAIN_S3_PATH
    aws_cli_agbm = f'aws s3 cp {agbm_s3_path} {raw_data_path} --recursive --no-sign-request'
    aws_cli_features_train = f'aws s3 cp {features_train_path} {raw_data_path} --recursive --no-sign-request'
    os.system(aws_cli_agbm)
    os.system(aws_cli_features_train)
    features_test_path = FEATURES_TEST_S3_PATH
    aws_cli_features_test = f'aws s3 cp {features_test_path} {raw_data_path} --recursive --no-sign-request'
    os.system(aws_cli_features_test)



def load_dataset():


    # Code to verify LOCAL_DATA_PATH folder existence and create in case it doesn't
    # LOAD_DATA_PATH is the path to download raw data files
    raw_data_path = LOCAL_DATA_PATH
    features_path = FEATURES_FILE
    agbm_s3_path = AGBM_S3_PATH

    check_data_path (raw_data_path)

    if not os.path.exists(os.path.expanduser(raw_data_path)):
        os.makedirs(os.path.expanduser(raw_data_path))
        print(Fore.BLUE + f'\nFolder {raw_data_path} was created\n' + Style.RESET_ALL)

    features = FEATURES_FILE
    mode = MODE
    month = MONTH
    chip_id_size = CHIP_ID_SIZE

    # Updates 'features_metadata.csv' in case download was interrupted before
    datafiles = os.listdir(os.path.expanduser(raw_data_path))
    datafiles_no_agbm = [item for item in datafiles if 'agbm' not in item]
    features['file_downloaded'] = features['filename'].isin(pd.Series(datafiles_no_agbm)).astype(bool)
    features.to_csv(os.path.expanduser(features_path), index = False)

    # Filter 'features_metadata' with mode and month
    featuresmonth = features_per_month (features, month)
    featuresmode = features_mode (featuresmonth, mode)
    features_to_download = features_not_downloaded (featuresmode)
    num_file = filters[month]

    # Download files according to 'features_to_download' dataframe
    if chip_id_size <= 20:
        chip_id_list = features_to_download['chip_id'].unique()[: chip_id_size]
        get_aws_chunk(features_to_download, raw_data_path,
                          agbm_s3_path, chip_id_list, num_file)
    elif chip_id_size >= 50:
        chip_id = np.asarray ([init + '*' for init in combs])
        get_aws_chunk(features_to_download, raw_data_path,
                          agbm_s3_path, chip_id, num_file)
    elif chip_id_size >= 1000:
        chip_id = np.asarray ([init + '*' for init in chip_id_letters])
        get_aws_chunk(features_to_download, raw_data_path,
                          agbm_s3_path, chip_id, num_file)

    # Updates 'features_metadata.csv' with newly downloaded data
    datafiles = os.listdir(os.path.expanduser(raw_data_path))
    datafiles_no_agbm = [item for item in datafiles if 'agbm' not in item]
    features['file_downloaded'] = features['filename'].isin(pd.Series(datafiles_no_agbm))
    features.to_csv(os.path.expanduser(features_path), index = False)
    print (Fore.GREEN + f"\n'features_metadata.csv' updated with downloaded files\n" + Style.RESET_ALL)



def preprocess():
    """
    This string takes a source from a set path and returns an array
    """
    print(f"\nPreprocessing data...")

    breakpoint()
    chip_ids_path = os.path.expanduser(f'{LOCAL_DATA_PATH}{MODE.capitalize()}/{chip_id_folder}')
    list_chip_ids = os.listdir(chip_ids_path)
    chip_ids = list_chip_ids[: int(len(list_chip_ids) * PERC)]
    filt_features = features_per_month(FEATURES_FILE, MONTH)
    filt_features = features_mode(filt_features, MODE)
    filt_features = features_downloaded(filt_features)
    filt_features = filt_features[filt_features['chip_id'] == chip_ids]

    SFiles = [file for file in filt_features['filename']]

    print(f"\n✅ Data processed saved entirely")
    prepro_data = np.asarray(0)
    return prepro_data


def train():
    """
    Train a new model on the full (already preprocessed) dataset ITERATIVELY, by loading it
    chunk-by-chunk, and updating the weight of the model after each chunks.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set
    common to all chunks.
    """
    print("\n⭐️ Use case: train")

    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Load a validation set common to all chunks, used to early stop model training
    #data_val_processed = get_chunk(source_name: str,
    #          index: int = 0,
    #          chunk_size: int = None,
    #          verbose=False)  # Retrieve all further data

    if data_val_processed is None:
        print("\n✅ no data to train")
        return None



    X_prepro_data = "file path here"
    y_prepro_data = "file path here"

    model = None
    model = load_model()  # production model

    # Model params
    learning_rate = 0.001
    batch_size = 256
    patience = 2

    # Iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0
    metrics_val_list = []

    while (True):

        print(Fore.BLUE + f"\nLoading and training on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_processed_chunk = get_chunk(
            source_name=f"train_processed_{DATASET_SIZE}",
            index=chunk_id * CHUNK_SIZE,
            chunk_size=CHUNK_SIZE
        )

        # Check whether data source contain more data
        if data_processed_chunk is None:
            print(Fore.BLUE + "\nNo more chunk data..." + Style.RESET_ALL)
            break

        data_processed_chunk = data_processed_chunk.to_numpy()

        X_train_chunk = data_processed_chunk[:, :-1]
        y_train_chunk = data_processed_chunk[:, -1]

        # Increment trained row count
        chunk_row_count = data_processed_chunk.shape[0]
        row_count += chunk_row_count

        # Initialize model
        if model is None:
            model = initialize_model(X_train_chunk)

        # (Re-)compile and train the model incrementally
        model = compile_model(model, learning_rate)
        model, history = train_model(
            model,
            X_train_chunk,
            y_train_chunk,
            batch_size=batch_size,
            patience=patience,
            validation_data=(X_val_processed, y_val)
        )

        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)
        print(f"Chunk MAE: {round(metrics_val_chunk,2)}")

        # Check if chunk was full
        if chunk_row_count < CHUNK_SIZE:
            print(Fore.BLUE + "\nNo more chunks..." + Style.RESET_ALL)
            break

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the training 👌")
        return

    # Return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    print(f"\n✅ trained on {row_count} rows with MAE: {round(val_mae, 2)}")

    params = dict(
        # Model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,

        # Package behavior
        context="train",
        chunk_size=CHUNK_SIZE,

        # Data source
        training_set_size=DATASET_SIZE,
        val_set_size=VALIDATION_DATASET_SIZE,
        row_count=row_count,
        model_version=get_model_version(),
    )

    # Save model
    save_model(model=model, params=params, metrics=dict(mae=val_mae))

    return val_mae


def evaluate():
    """
    Evaluate the performance of the latest production model on new data
    """

    print("\n⭐️ Use case: evaluate")

    # Load new data
    new_data = get_chunk(
        source_name=f"val_processed_{DATASET_SIZE}",
        index=0,
        chunk_size=None
    )  # Retrieve all further data

    if new_data is None:
        print("\n✅ No data to evaluate")
        return None

    new_data = new_data.to_numpy()

    X_new = new_data[:, :-1]
    y_new = new_data[:, -1]

    model = load_model()

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    mae = metrics_dict["mae"]

    # Save evaluation
    params = dict(
        dataset_timestamp=get_dataset_timestamp(),
        model_version=get_model_version(),

        # Package behavior
        context="evaluate",

        # Data source
        training_set_size=DATASET_SIZE,
        val_set_size=VALIDATION_DATASET_SIZE,
        row_count=len(X_new)
    )

    save_model(params=params, metrics=dict(mae=mae))

    return mae


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    from taxifare.ml_logic.registry import load_model

    if X_pred is None:

        X_pred = pd.DataFrame(dict(
            key=["2013-07-06 17:18:00"],  # useless but the pipeline requires it
            pickup_datetime=["2013-07-06 17:18:00 UTC"],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1]
        ))

    model = load_model()

    X_processed = preprocess_features(X_pred)

    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape)

    return y_pred


if __name__ == '__main__':
    preprocess()
    train()
    pred()
    evaluate()
