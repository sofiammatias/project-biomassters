import numpy as np
import pandas as pd
import os

from colorama import Fore, Style

from biomassters.data_sources.aws import get_aws_chunk, features_per_month, features_mode
from biomassters.data_sources.aws import features_not_downloaded

#from taxifare.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
#from taxifare.ml_logic.params import CHUNK_SIZE, DATASET_SIZE, VALIDATION_DATASET_SIZE
#from taxifare.ml_logic.preprocessor import preprocess_features
#from taxifare.ml_logic.utils import get_dataset_timestamp
#from taxifare.ml_logic.registry import get_model_version

#from biomassters.ml_logic.registry import load_model, save_model

def load_all_dataset():
    raw_data_path = '~/.project-biomassters/raw_data/'
    agbm_s3_path = 's3://drivendata-competition-biomassters-public-us/train_agbm/'
    aws_cli_agbm = f'aws s3 cp {agbm_s3_path} {raw_data_path} --recursive --no-sign-request'
    os.system(aws_cli_agbm)
    features_train_path='s3://drivendata-competition-biomassters-public-us/train_features/'
    aws_cli_features_train = f'aws s3 cp {features_train_path} {raw_data_path} --recursive --no-sign-request'
    os.system(aws_cli_features_train)
    features_test_path='s3://drivendata-competition-biomassters-public-us/test_features/'
    aws_cli_features_test = f'aws s3 cp {features_test_path} {raw_data_path} --recursive --no-sign-request'
    os.system(aws_cli_features_test)



def load_dataset():


    # Code to verify LOCAL_DATA_PATH folder existence and create in case it doesn't
    # LOAL_DATA_PATH is the path to download raw data files
    raw_data_path = os.getenv('LOCAL_DATA_PATH')
    features_path = os.getenv('FEATURES')
    agbm_s3_path = os.getenv('TRAIN_AGBM_S3_PATH')

    if not os.path.exists(raw_data_path):
        os.makedirs (raw_data_path)
        print (Fore.BLUE + f'\nFolder {raw_data_path} was created\n' + Style.RESET_ALL)

    # Loading remaining data from env variables
    # 'features_metadata'
    features = pd.read_csv(os.getenv('FEATURES'))
    # define usage mode: train or test
    mode = os.getenv('MODE')
    # set month to download files
    month = os.getenv('MONTH')
    # set 'chunks' of chip_id
    chip_id_num = int(os.getenv('CHIP_ID_NUM'))
    # set the number of chip_id's for each 'chunk': 'chunk' size
    chip_id_size = int(os.getenv('CHIP_ID_SIZE'))


    # Filter 'features_metadata' with mode and month
    #featuresmonth = features_per_month (features, month)
    #print (featuresmonth.shape)
    #featuresmode = features_mode (featuresmonth, mode)
    #features_to_download = features_not_downloaded (featuresmode, mode)
    features_to_download = features


    # Download files according to 'features_to_download' dataframe
    if chip_id_size <= 20:
        if chip_id_num > 10:
            chip_id_num = 10
        chip_id_list = features_to_download['chip_id'].unique()[: chip_id_num*chip_id_size]
        for filter in chip_id_list:
            print (filter)
            get_aws_chunk(features_to_download, raw_data_path, features_path,
                          agbm_s3_path, filter)


    else:
        pass
        #if chip_id_num > 5:
        #    chip_id_num = 5
        #letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        #           'a', 'b', 'c', 'd', 'e', 'f']

        #get_aws_chunk(features_to_download, raw_data_path)



def preprocess(source_type = 'train'):
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """

    print("\n⭐️ Use case: preprocess")

    # Iterate on the dataset, in chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0
    source_name = f"{source_type}_{DATASET_SIZE}"
    destination_name = f"{source_type}_processed_{DATASET_SIZE}"

    while (True):
        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_chunk = get_chunk(
            source_name=source_name,
            index=chunk_id * CHUNK_SIZE,
            chunk_size=CHUNK_SIZE
        )

        # Break out of while loop if data is none
        if data_chunk is None:
            print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
            break

        row_count += data_chunk.shape[0]

        data_chunk_cleaned = clean_data(data_chunk)

        cleaned_row_count += len(data_chunk_cleaned)

        # Break out of while loop if cleaning removed all rows
        if len(data_chunk_cleaned) == 0:
            print(Fore.BLUE + "\nNo cleaned data in latest chunk..." + Style.RESET_ALL)
            break

        X_chunk = data_chunk_cleaned.drop("fare_amount", axis=1)
        y_chunk = data_chunk_cleaned[["fare_amount"]]

        X_processed_chunk = preprocess_features(X_chunk)

        data_processed_chunk = pd.DataFrame(
            np.concatenate((X_processed_chunk, y_chunk), axis=1)
        )

        # Save and append the chunk
        is_first = chunk_id == 0

        save_chunk(
            destination_name=destination_name,
            is_first=is_first,
            data=data_processed_chunk
        )

        chunk_id += 1

    if row_count == 0:
        print("\n✅ No new data for the preprocessing 👌")
        return None

    print(f"\n✅ Data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")

    return None

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
    data_val_processed = get_chunk(
        source_name=f"val_processed_{VALIDATION_DATASET_SIZE}",
        index=0,  # retrieve from first row
        chunk_size=None
    )  # Retrieve all further data

    if data_val_processed is None:
        print("\n✅ no data to train")
        return None

    data_val_processed = data_val_processed.to_numpy()

    X_val_processed = data_val_processed[:, :-1]
    y_val = data_val_processed[:, -1]

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
        dataset_timestamp=get_dataset_timestamp(),
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
    preprocess(source_type='val')
    train()
    pred()
    evaluate()
