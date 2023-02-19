import os
import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

import utils
from models.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--train_path")
parser.add_argument("--val_path")

if __name__ == "__main__":

    args = parser.parse_args()

    
    # Load the config model
    config = utils.load_config(
        config_path='../config.yml'
    )

    # Set logger
    utils.set_logger(os.path.join(config['checkpoint_dir'], 'train.log'))
    logging.info("Loading dataset...")
    # Load data for training
    df_train = pd.read_csv(args.train_path)
    if args.val_path != None:
        df_val = pd.read_csv(args.val_path)
    else:
        df_train, df_val = train_test_split(
            df_train,
            test_size=0.2,
            random_state=100
        )
    
    logging.info("Done!\n")

    # labels = sorted(set(df_train['category']))
    # label2idx = dict(zip(labels, range(len(labels))))
    # trainer = Trainer(
    #     config=config,
    #     df_train=df_train,
    #     df_val=df_val,
    #     label2idx=label2idx
    # )

    # trainer.train()