import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

class MLP(nn.Module):

    def __init__(self, config):
        super(MLP, self).__init__()
        self.label_size = config['label_size']
        self.hidden_size = config['hidden_size']
        self.embedding_dim = config['embedding_dim']
        self.dropout_prob = config['dropout']

        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer1 = nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.embedding_dim)
        )
        self.b1 = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.layer2 = nn.Parameter(
            torch.FloatTensor(self.label_size, self.hidden_size)
        )
        self.b2 = nn.Parameter(torch.FloatTensor(self.label_size))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters
        :return:
        """
        std_v1 = 1.0 / math.sqrt(self.hidden_size)
        self.layer1.data.uniform_(-std_v1, std_v1)
        self.b1.data.uniform_(-std_v1, std_v1)

        std_v2 = 1.0 / math.sqrt(self.label_size)
        self.layer2.data.uniform_(-std_v2, std_v2)
        self.b2.data.uniform_(-std_v2, std_v2)

    def forward(self, embedding_text):
        layer1_output = F.relu(F.linear(embedding_text, self.layer1, self.b1))
        layer1_output = self.dropout(layer1_output)
        logit = F.linear(layer1_output, self.layer2, self.b2)
        return logit



# class Transformer(nn.Module):

#     def __init__(self, pretrain, dropout):

#         super(Transformer, self).__init__()

#         self.pretrain = AutoModel.from_pretrained(pretrain)
#         self.dropout = nn.Dropout(dropout)
#         self.linear1 = nn.Linear(768, 256)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(256, 5)
#         self.sm = nn.Softmax()

#     def forward(self, input_id, mask):

#         _, pooled_output = self.pretrain(
#             input_ids= input_id,
#             attention_mask=mask,
#             return_dict=False
#         )
#         dropout_output = self.dropout(pooled_output)
#         hidden_state1 = self.linear1(dropout_output)
#         hidden_state2 = self.relu(hidden_state1)
#         hidden_state3 = self.linear2(hidden_state2)

#         final_layer = self.sm(hidden_state3)

#         return final_layer