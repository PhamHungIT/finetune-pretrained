from torch import nn
from transformers import AutoModel

class Encoder(nn.Module):

    def __init__(self, pretrain, dropout):

        super(Encoder, self).__init__()

        self.pretrain = AutoModel.from_pretrained(pretrain)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 5)
        self.sm = nn.Softmax()

    def forward(self, input_id, mask):

        _, pooled_output = self.pretrain(
            input_ids= input_id,
            attention_mask=mask,
            return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        hidden_state1 = self.linear1(dropout_output)
        hidden_state2 = self.relu(hidden_state1)
        hidden_state3 = self.linear2(hidden_state2)

        final_layer = self.sm(hidden_state3)

        return final_layer