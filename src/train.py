import os
import argparse
import logging

import torch
from torch.utils.data import DataLoader

import utils
from data.dataset import Dataset
from data.embedding import Embedding

from models.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--train_file")
parser.add_argument("--val_file")
parser.add_argument("--text_col")
parser.add_argument("--label_col")
parser.add_argument("--checkpoint_dir")
parser.add_argument("--checkpoint_path")
parser.add_argument("--vectorizer")

args = parser.parse_args()

# Load the config model
config = utils.load_config(
    config_path='../config.yml'
)
model_config = config['model']
model_config['checkpoint_dir'] = args.checkpoint_dir
# Set logger
log_path = os.path.join(model_config['checkpoint_dir'], 'train.log')
utils.set_logger(log_path=log_path)



def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
         label_list.append(label2idx[_label])
         processed_text = torch.tensor(embedding(_text), dtype=torch.float32)
         text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.row_stack(text_list)
    return label_list.to(device), text_list.to(device)


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load data, clean, split data train whether not contain valid data
    logging.info("Loading dataset...")
    df_train, df_val = utils.load_csv(
        train_file=args.train_file,
        val_file=args.val_file,
        text_col=args.text_col,
        label_col=args.label_col
    )
    logging.info("Done!\n")

    # Create vectorizer, label encoder
    embedding = Embedding(type=args.vectorizer)
    embedding.fit(
        corpus=df_train[args.text_col].tolist()
    )
    model_config['embedding_dim'] = embedding.embedding_dim
    model_config['len_train'] = len(df_train)
    model_config['len_val'] = len(df_val)
    
    logging.info("Vectorizer: {}".format(args.vectorizer))
    logging.info("Embedding dimension: {}".format(embedding.embedding_dim))
    labels = sorted(set(df_train[args.label_col]))
    label2idx = dict(zip(labels, range(len(labels))))

    # Create dataloader
    train_dataset = Dataset(
        df=df_train,
        text_col=args.text_col,
        label_col=args.label_col
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_batch,
        batch_size=256
        # batch_size=config['batch_size']
    )

    valid_dataset = Dataset(
        df=df_val,
        text_col=args.text_col,
        label_col=args.label_col    
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=True,
        collate_fn=collate_batch,
        batch_size=256
        # batch_size=config['batch_size']
    )

    # if args.checkpoint_path != None:
    #     # Training from old checkpoint
    #     state = torch.load(args.checkpoint_path, map_location=device)
    #     trainer = Trainer(
    #         label2idx=state['label2idx'],
    #         config=model_config
    #     )
    #     trainer.encoder.load_state_dict(state['state_dict'])

    # else:
    #     trainer = Trainer(
    #         config=model_config,
    #         label2idx=label2idx
    #     )

    trainer = Trainer(
        label2idx=label2idx,
        config=model_config
    )
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader
    )
