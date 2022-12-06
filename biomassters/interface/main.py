import numpy as np
import pandas as pd
import os

from colorama import Fore, Style

from biomassters.data_sources.aws import get_aws_chunk
from biomassters.data_sources.utils import features_not_downloaded, check_for_downloaded_files
from biomassters.data_sources.utils import check_data_path, organize_folders
from biomassters.data_sources.utils import features_per_month, features_mode
from biomassters.data_sources.utils import set_trained_files, set_predicted_chip_ids
from biomassters.ml_logic.params import LOCAL_DATA_PATH, FEATURES_FILE, MODE, MONTH
from biomassters.ml_logic.params import AGBM_S3_PATH ,FEATURES_TRAIN_S3_PATH
from biomassters.ml_logic.params import FEATURES_TEST_S3_PATH, CHIP_ID_SIZE
from biomassters.ml_logic.params import LOCAL_OUTPUT_PATH
from biomassters.ml_logic.params import filters, chip_id_letters, combs
from biomassters.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from biomassters.ml_logic.data import import_data, get_predictions, create_chip_ids_list
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
    """
    Verifies LOCAL_DATA_PATH folder existence and creates it in case it doesn't exist
    """
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

    if features['file_downloaded'] is None:
        features['file_downloaded'] = False
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




def organizing_data():
    """
    This string takes a source from a set path and returns an array
    """
    print(f"\n⭐️ Use case: organize data")
    print(f"\nOrganizing data folders and checking downloaded files for errors...")

    organize_folders()
    check_for_downloaded_files()

    print(f"\n✅ Data is ready for processing")


def train():
    """
    Train a new model on the full dataset, by loading it chunk-by-chunk by chip_id
    size, and updating the weight of the model after each chunks.
    Save final model once it has seen all data, and compute validation metrics
    common to all chunks.
    """
    i = 0

    print(f"\n⭐️ Use case: train")

    print(Fore.BLUE + "\nLoading data..." + Style.RESET_ALL)

    # Load a validation set common to all chunks, used to early stop model training
    #data_val_processed = get_chunk(source_name: str,
    #          index: int = 0,
    #          chunk_size: int = None,
    #          verbose=False)  # Retrieve all further data

    if not os.path.exists(f'{LOCAL_DATA_PATH}{MODE.capitalize()}'):
        print(Fore.RED + "\nData is not ready for processing. Run run_organizing_data function first." + Style.RESET_ALL)
        return None

    chip_ids = create_chip_ids_list()

    while i <= len(chip_ids):

        X1, X2, y = import_data(i, chip_ids)

        if  (X1 is None) and (X2 is None) and (y is None):
            print(Fore.RED + "\nData is not ready for processing. Run run_organizing_data function first or download files to continue." + Style.RESET_ALL)
            return None

        if i in range(0, 8000, 1000):
            model = None
            model = load_model()  # production model

            # Iterate on the full dataset per chunks
            metrics_val_list = []

            # Initialize model
            if model is None:
                model = initialize_model(32)

            # (Re-)compile and train the model incrementally
            model = compile_model(model)

        print(f"Training on chip {chip_ids[i]}.")
        model, history = train_model(model, X1, X2, y)

        metrics_val_chunk = np.min(history.history['root_mean_squared_error'])
        metrics_val_list.append(metrics_val_chunk)
        rmse = metrics_val_list[-1]

        print(f"Chunk RMSE: {round(metrics_val_chunk,2)}")
        print(f"Done with chip {chip_ids[i]}!")

        if i in range(0, 8000, 1000):
            # Return the last value of the validation MAE
            rmse = metrics_val_list[-1]

            params = dict(
                # Model parameters
                chip_id_size = len(chip_ids),
                # Package behavior
                context='train',
                # Data source
               training_set_size=len(X1),
                model_version=get_model_version(),
            )

            # Save model
            save_model(model=model, params=params, metrics=dict(rmse=rmse))

        i += 1

        if i == len(chip_ids):

            params = dict(
                # Model parameters
                chip_id_size = len(chip_ids),
                # Package behavior
                context='train',
                # Data source
                training_set_size=len(X1),
                model_version=get_model_version(),
                )

            # Save model
            save_model(model=model, params=params, metrics=dict(rmse=rmse))
            print(f"\n✅All {len(chip_ids)} chips have been trained with RMSE: {round(rmse, 2)} 👌!")
            break

    return model, history



def evaluate():
    """
    Evaluate the performance of the latest production model on new data
    """

    print("\n⭐️ Use case: evaluate")

    # Load data
    X1, X2, y, chip_ids_list = import_data()

    model = load_model()

    for x, _ in enumerate (chip_ids_list):
        metrics_dict = evaluate_model(model=model, X1=X1[x], X2=X2[x], y=y[x])
        mse = metrics_dict["mse"]

    # Save evaluation
    params = dict(
    #    dataset_timestamp=get_dataset_timestamp(),
        model_version=get_model_version(),

        # Package behavior
        context="evaluate",

        # Data source
        training_set_size=len(X1),
    )

    save_model(params=params, metrics=dict(mse=mse))

    set_trained_files(chip_ids_list)

    return mse

def pred() -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    os.environ['MODE'] = 'test'
    MODE = 'test'

    model = load_model()
    y_pred, chip_ids_list = get_predictions (model)
    set_predicted_chip_ids(chip_ids_list)

    print(f"\n✅ Predictions saved in {LOCAL_OUTPUT_PATH} ")

    return y_pred


if __name__ == '__main__':
    organizing_data()
    train()
    pred()
    evaluate()
