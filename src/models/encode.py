from torch import nn
from transformers import AutoModel

class Encode(nn.Module):

    def __init__(self, pretrain, dropout=0.5):

        super(Encode, self).__init__()

        self.pretrain = AutoModel.from_pretrained(pretrain)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.pretrain(
            input_ids= input_id,
            attention_mask=mask,
            return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer