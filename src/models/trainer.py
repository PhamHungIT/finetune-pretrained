import os
import torch
import logging
import shutil

from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from transformers import AutoTokenizer

from models.encoder import MLP

# logging.getLogger(__name__)

class Trainer:
    
    def __init__(self, config) -> None:
                
        """ Model config """

        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

        self.config = config
        self.encoder = MLP(
            config=config
        )
        self.encoder = torch.nn.DataParallel(self.encoder)

    def train(self, train_dataloader, val_dataloader, label2idx):
        
        
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Training on GPU: {use_cuda} - Device: {device}")
        
        
        self.label2idx = label2idx
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.encoder.parameters(), lr=self.learning_rate)
        # scheduler = lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1
        )
        if use_cuda:
                self.encoder = self.encoder.cuda()
                criterion = criterion.cuda()
        
        best_acc_val = 0
        
        for epoch_num in range(self.epochs):
            self.encoder.train()
            total_acc_train = 0
            total_loss_train = 0

            for label_ids, text_embeddings in tqdm(
                train_dataloader,
                colour='green',
            ):
                optimizer.zero_grad()
                predicted_labels = self.encoder(text_embeddings)
                loss = criterion(predicted_labels, label_ids)
                total_loss_train += loss.item()
                
                acc = (predicted_labels.argmax(1) == label_ids).sum().item()
                total_acc_train += acc
                
                loss.backward()
                optimizer.step()
                scheduler.step()


            """ Validate """
            self.encoder.eval()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():
                for label_ids, text_embeddings  in val_dataloader:

                    predicted_labels = self.encoder(text_embeddings)
                    loss = criterion(predicted_labels, label_ids)
                    total_loss_val += loss.item()
                    
                    acc = (predicted_labels.argmax(1) == label_ids).sum().item()
                    total_acc_val += acc
            
            loss_train = total_loss_train / self.config['len_train']
            acc_train = total_acc_train / self.config['len_train']
            loss_val = total_loss_val / self.config['len_val']
            acc_val = total_acc_val / self.config['len_val']

            logging.info(
                f'Epochs: {epoch_num + 1} | Train Loss: {loss_train: .3f} \
                | Train Accuracy: {acc_train: .3f} \
                | Val Loss: {loss_val: .3f} \
                | Val Accuracy: {acc_val: .3f}')
            
            """Save checkpoints"""
            if acc_val > best_acc_val:
                logging.info("Found better model!")
                self.save_checkpoint(is_best=True)
        self.save_checkpoint(is_best=False)

    def save_checkpoint(self, is_best):
        state = {
            "config": self.config,
            "label2idx": self.label2idx,
            "state_dict": self.encoder.state_dict()
        }

        checkpoint_dir = self.config['checkpoint_dir']
        file_path = os.path.join(checkpoint_dir, "last.pt")
        with open(file_path, 'wb') as f:
            torch.save(state, f, _use_new_zipfile_serialization=False)
        
        if is_best:
            shutil.copyfile(file_path, os.path.join(checkpoint_dir, 'best.pt'))
