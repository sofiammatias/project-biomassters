
import pandas as pd

def features_per_month (features:pd.DataFrame, month:str) -> pd.DataFrame:
    """
    Filter 'features_metadata' for one month and return the filtered dataframe
    """
    return features[features.month == month]

def features_mode (features:pd.DataFrame, mode:str) -> pd.DataFrame:
    """
    Filter 'features_metadata' per using mode: train or test (uses 'split' column)
    """
    return features[features.split == mode.lower()]

def features_not_downloaded (features:pd.DataFrame) -> pd.DataFrame:
    """
    Filter 'features_metadata' per using mode: train or test (uses 'split' column)
    """
    return features[features.file_downloaded == False]
