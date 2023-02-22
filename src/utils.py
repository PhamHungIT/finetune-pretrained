import logging

import yaml
import pandas as pd


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


def load_csv(csv_file, col_text, col_category, sep):
    """Load csv file, convert data to dictionary

    Args:
        csv_file: path to csv data
        col_text: column containing sentences
        col_category: column containing labels
        sep: delimiter between 2 columns
    """

    df = pd.read_csv(csv_file, sep=sep)
    msg = "Not exist columns {} and {} in your data!".format(
        col_text,
        col_category
    )
    assert col_text in list(df.columns) and col_category in list(df.columns), msg
    data = {
        "text": list(df[col_text].values),
        "category": list(df[col_category].values)
    }
    return data