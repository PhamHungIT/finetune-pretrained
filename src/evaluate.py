import argparse

import pandas as pd
import torch

import utils
utils.set_logger(
    log_path='../checkpoints/evaluate.log'
)

from src.models.old_trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--test_path")
parser.add_argument("--checkpoint_path")


if __name__ == "__main__":
    args = parser.parse_args()

    # Load data for testing
    df_test = pd.read_csv(args.test_path)
    print(len(df_test))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    state = torch.load(args.checkpoint_path, map_location=device)
    trainer = Trainer(
        label2idx=state['label2idx'],
        config=state['config']
    )

    trainer.encoder.load_state_dict(state['state_dict'])
    trainer.evaluate(df_test=df_test)