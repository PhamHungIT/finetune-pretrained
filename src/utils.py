import logging

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas(desc="Preprocessing")

from data.preprocess import clean_text

def load_config(config_path):
    with open(config_path, 'r') as fi:
        config = yaml.safe_load(fi)
    return config

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def load_csv(
        train_file,
        val_file=None,
        text_col='text',
        label_col='category',
        sep='\t'
    ):
    """Load, split data

    Args:
        train_file: path to csv data train
        val_file: path to csv data validate
        text_col: column containing sentences
        col_category: column containing labels
        sep: delimiter between 2 columns
    """
    df_train = pd.read_csv(train_file, sep=sep)
    df_train = df_train[[text_col, label_col]]
    df_train.dropna(inplace=True)
    if val_file != None:
        df_val = pd.read_csv(val_file, sep=sep)
        df_val.dropna(inplace=True)
    else:
        df_train, df_val = train_test_split(
            df_train,
            test_size=0.2,
            random_state=100
        )

    df_train[text_col] = df_train[text_col].progress_apply(clean_text)
    df_train.drop_duplicates(subset=text_col, inplace=True)
    
    df_val[text_col] = df_val[text_col].progress_apply(clean_text)
    df_val.drop_duplicates(subset=text_col, inplace=True)

    return df_train, df_val