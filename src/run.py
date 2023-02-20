import os
import argparse
import logging

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--train_path")
parser.add_argument("--val_path")
parser.add_argument("--checkpoint_dir")
parser.add_argument("--checkpoint_path")

args = parser.parse_args()

# Load the config model
config = utils.load_config(
    config_path='../config.yml'
)

config['checkpoint_dir'] = args.checkpoint_dir
# Set logger
log_path = os.path.join(config['checkpoint_dir'], 'train.log')
utils.set_logger(log_path=log_path)

from models.trainer import Trainer


if __name__ == "__main__":

    # Load data for training
    logging.info("Loading dataset...")
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

    labels = sorted(set(df_train['category']))
    label2idx = dict(zip(labels, range(len(labels))))

    if args.checkpoint_path != None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        state = torch.load(args.checkpoint_path, map_location=device)
        trainer = Trainer(
            label2idx=state['label2idx'],
            config=config
        )
        trainer.encoder.load_state_dict(state['state_dict'])
    else:
        trainer = Trainer(
            config=config,
            label2idx=label2idx
        )

    trainer.train(
        df_train=df_train,
        df_val=df_val
    )
