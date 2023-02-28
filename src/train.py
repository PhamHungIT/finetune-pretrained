import os
import argparse
import logging

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--train_path")
parser.add_argument("--val_path")
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

from models.trainer import Trainer





if __name__ == "__main__":

    # Load data, split data train whether not contain valid data
    logging.info("Loading dataset...")
    df_train = pd.read_csv(args.train_path)
    df_train.dropna(inplace=True)
    if args.val_path != None:
        df_val = pd.read_csv(args.val_path)
        df_val.dropna(inplace=True)
    else:
        df_train, df_val = train_test_split(
            df_train,
            test_size=0.2,
            random_state=100
        )
        
    logging.info("Done!\n")



    if args.vectorizer == 'tf-idf':
        tf_idf = TfidfVectorizer()
        vectorizer = tf_idf.fit(df_train['text'].tolist())
    
    elif args.vectorizer == 'bow':
        bow = CountVectorizer()
        vectorizer = bow.fit(df_train['text'].tolist())
    
    elif args.vectorizer == 'w2v':
        w2v_config = config['word2vec']
        split_sentences = [sentence.split() for sentence in df_train['text']]
        vectorizer = Word2Vec(
            sentences=split_sentences,
            vector_size=w2v_config['vector_size'],
            min_count=w2v_config['min_count'],
            workers=os.cpu_count() - 1
        )

    if args.vectorizer != 'w2v':
        text_pipeline = lambda text: vectorizer.transform([text]).toarray()[0]
    else:
        text_pipeline = lambda text: vectorizer.wv[text]
    label_pipeline = lambda label: label2idx[label]
    
    
    
    labels = sorted(set(df_train['category']))
    label2idx = dict(zip(labels, range(len(labels))))


    if args.checkpoint_path != None:
        # Training from old checkpoint
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        state = torch.load(args.checkpoint_path, map_location=device)
        trainer = Trainer(
            label2idx=state['label2idx'],
            config=model_config
        )
        trainer.encoder.load_state_dict(state['state_dict'])
    else:
        trainer = Trainer(
            config=model_config,
            label2idx=label2idx
        )

    trainer.train(
        df_train=df_train,
        df_val=df_val
    )
