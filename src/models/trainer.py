import os
import torch
import logging
import shutil

from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from transformers import AutoTokenizer

from models.encoder import Encoder
from data.dataset import Dataset


class Trainer:
    
    def __init__(self, label2idx, config) -> None:
                
        """ Model config """
        self.encoder = Encoder(
            pretrain=config['pretrain'],
            dropout=config['dropout']
        )
        self.encoder = torch.nn.DataParallel(self.encoder)

        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

        self.label2idx = label2idx
        self.tokenizer = AutoTokenizer.from_pretrained(config['pretrain'])
        self.config = config
    
    def train(self, df_train, df_val):
        logging.info("\nCreate dataloader")
        train = Dataset(
            df=df_train,
            label2idx=self.label2idx,
            tokenizer=self.tokenizer            
        )
        
        val = Dataset(
            df = df_val,
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
        logging.info(
            "\nConfig:\n\t" +
            str('\n\t'.join("{}: {}".format(str(k), str(v)) for k, v in self.config.items()))
        )

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Training on GPU: {use_cuda} - Device: {device}")
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
        
        self.encoder.train()
        best_acc_val = 0
        for epoch_num in range(self.epochs):

            self.encoder.train()

            total_acc_train = 0
            total_loss_train = 0
            for train_input, train_label in tqdm(
                train_dataloader,
                colour='green',
            ):
                optimizer.zero_grad()
                
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                output = self.encoder(input_id, mask)
                
                # batch_loss = criterion(output, train_label.long())
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                batch_loss.backward()
                optimizer.step()
                
                scheduler.step()
            

            """ Validate """
            self.encoder.eval()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = self.encoder(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            loss_train = total_loss_train / len(train)
            acc_train = total_acc_train / len(train)
            loss_val = total_loss_val / len(val)
            acc_val = total_acc_val / len(val)

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

    def evaluate(self, df_test):

        test = Dataset(
            df = df_test,
            label2idx=self.label2idx,
            tokenizer=self.tokenizer 
        )

        test_dataloader = torch.utils.data.DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=True
        )

        logging.info(f"Data test: {len(test)}")
        logging.info(
            "\nConfig:\n\t" +
            str('\n\t'.join("{}: {}".format(str(k), str(v)) for k, v in self.config.items()))
        )

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Inferring on GPU: {use_cuda} - Device: {device}")
        if use_cuda:
            self.encoder = self.encoder.cuda()

        self.encoder.eval()
            
        total_acc_val = 0
        with torch.no_grad():

            for test_input, test_label in tqdm(test_dataloader):

                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = self.encoder(input_id, mask)
                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_val += acc
        
        acc_test = total_acc_val / len(test)
        logging.info(f"Accuracy test is {str(acc_test)}")


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
