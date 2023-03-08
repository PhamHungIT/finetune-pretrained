
import argparse
import os
import time

import pandas as pd
import torch
from tqdm import tqdm
from tqdm import trange
from sklearn.metrics import classification_report

from data.embedding import Embedding
from data.preprocess import clean_text
from models.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--test_path")
parser.add_argument("--vectorizer")
parser.add_argument("--checkpoint_dir")
parser.add_argument("--text_col")
parser.add_argument("--category_col")



class Infer:

    def __init__(self, embedding_type, embedding_path, model_path):

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.embedding = Embedding(type=embedding_type)
        self.embedding.load(path=embedding_path)

        state = torch.load(model_path, map_location=self.device)
        config = state["config"]
        self.idx2label = {v:k for k,v in state["label2idx"].items()}

        self.model = Trainer(config=config)
        self.model.encoder.load_state_dict(state["state_dict"])



    def __call__(
        self,
        texts,
        batch_size,

    ):
        progress_bar = tqdm(texts, desc="Preprocess", ncols=100)
        clean_texts = [clean_text(s) for s in progress_bar]
        embedding_texts = [torch.tensor(self.embedding(s), dtype=torch.float32)
                           for s in clean_texts]
        pred_labels = []
        pred_probs = []
        with torch.no_grad():
            self.model.encoder.eval()
            for i in trange(0, len(texts), batch_size, colour='green', desc="Inferring", ncols=100):
            
                
                batch_texts = embedding_texts[i: i + batch_size]
                batch_texts = torch.row_stack(batch_texts)
                batch_texts.to(self.device)
                
                output = self.model.encoder(batch_texts)
                output = torch.softmax(output, dim=-1)
                
                cur_probs, cur_preds = torch.max(output, dim=-1)

                pred_labels.extend([self.idx2label[pred.item()] for pred in cur_preds])
                pred_probs.extend([cur_prob.item() for cur_prob in cur_probs])
        
        return pred_labels, pred_probs

if __name__ == "__main__":

    args = parser.parse_args()

    if args.vectorizer == "word2vec":
        embedding_path = os.path.join(args.checkpoint_dir, 'w2v.model')
    else:
        embedding_path = os.path.join(args.checkpoint_dir, "{}.pickle".format(args.vectorizer))
    
    model_path = os.path.join(args.checkpoint_dir, "best.pt")
    infer = Infer(
        embedding_path=embedding_path,
        embedding_type=args.vectorizer,
        model_path=model_path,
    )

    
    df_test = pd.read_csv(args.test_path, sep='\t')
    texts = df_test[args.text_col].tolist()

    s_time = time.time()
    y_pred, _ = infer(texts,batch_size=128)
    print("Time infer: {}".format((time.time() - s_time)* 1000 / len(df_test)))
    
    y_true = df_test[args.category_col].tolist()
    print(classification_report(y_true, y_pred))