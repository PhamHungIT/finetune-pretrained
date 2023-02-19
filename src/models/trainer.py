
import torch
import logging

from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from transformers import AutoTokenizer

import utils
from models.encode import Encode
from data.dataset import Dataset


class Trainer:
    
    def __init__(self, df_train, df_val, label2idx, config) -> None:
        
        """ Data """
        self.train_data = df_train
        self.val_data = df_val
        
        """ Model config """
        self.encode = Encode(
            pretrain=config['pretrain'],
            dropout=config['dropout']
        )

        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

        self.label2idx = label2idx
        self.tokenizer = AutoTokenizer.from_pretrained(config['pretrain'])

    
    def train(self):
        logging.info("Create dataloader")
        train = Dataset(
            df=self.train_data,
            label2idx=self.label2idx,
            tokenizer=self.tokenizer            
        )
        
        val = Dataset(
            df = self.val_data,
            label2idx=self.label2idx,
            tokenizer=self.tokenizer 
        )

        train_dataloader = torch.utils.data.DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val,
            batch_size=self.batch_size
        )
        logging.info(f"Data train: {len(train)}")
        logging.info(f"Data val: {len(val)}")
        logging.info("Done!")
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Training on GPU: {use_cuda} - Device: {device}")
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.encode.parameters(), lr=self.learning_rate)

        # scheduler = lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1
        )
        if use_cuda:

                self.encode = self.encode.cuda()
                criterion = criterion.cuda()
        
        self.encode.train()
        
        for epoch_num in range(self.epochs):

                self.encode.train()

                total_acc_train = 0
                total_loss_train = 0
                for train_input, train_label in tqdm(
                    train_dataloader,
                    colour='green',
                    desc=f"Epoch: {epoch_num + 1}/{self.epochs}"
                ):
                    optimizer.zero_grad()
                    
                    train_label = train_label.to(device)
                    mask = train_input['attention_mask'].to(device)
                    input_id = train_input['input_ids'].squeeze(1).to(device)
                    output = self.encode(input_id, mask)
                    
                    # batch_loss = criterion(output, train_label.long())
                    batch_loss = criterion(output, train_label)
                    total_loss_train += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == train_label).sum().item()
                    total_acc_train += acc

                    batch_loss.backward()
                    optimizer.step()
                    
                    scheduler.step()
                

                """ Validate """
                self.encode.eval()
                
                total_acc_val = 0
                total_loss_val = 0

                with torch.no_grad():

                    for val_input, val_label in val_dataloader:

                        val_label = val_label.to(device)
                        mask = val_input['attention_mask'].to(device)
                        input_id = val_input['input_ids'].squeeze(1).to(device)

                        output = self.encode(input_id, mask)

                        batch_loss = criterion(output, val_label)
                        total_loss_val += batch_loss.item()
                        
                        acc = (output.argmax(dim=1) == val_label).sum().item()
                        total_acc_val += acc
                
                loss_train = total_loss_train / len(self.train_data)
                acc_train = total_acc_train / len(self.train_data)
                loss_val = total_loss_val / len(self.val_data)
                acc_val = total_acc_val / len(self.val_data)

                logging.info(
                    f'Epochs: {epoch_num + 1} | Train Loss: {loss_train: .3f} \
                    | Train Accuracy: {acc_train: .3f} \
                    | Val Loss: {loss_val: .3f} \
                    | Val Accuracy: {acc_val: .3f}')
        