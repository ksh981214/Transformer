import torch
import torch.nn as nn
import math

from config import config
from sublayer import FeedForward,MultiHeadAttention

config = config()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        '''
            MultiHeadAttention: 1
            Add & LayerNorm: 2
            FeedForward: 1 
        '''
        self.batch_size = config.batch_size
        self.max_len=config.max_len
        self.model_dim = config.model_dim
        self.p_drop = config.p_drop

        self.dropout = nn.Dropout(self.p_drop)

        self.mha = MultiHeadAttention()
        self.ann1 = nn.LayerNorm([self.batch_size, self.max_len, self.model_dim])

        self.ff = FeedForward()
        self.ann2 = nn.LayerNorm([self.batch_size, self.max_len, self.model_dim])

    def forward(self, inputs):
        '''
            MultiHeadAttention --> Add&LayerNorm --> FeedForward --> Add&LayerNorm
        '''
        x = inputs.clone()
        output = self.mha(inputs, masking=False, encoded_output=None)
        output = self.dropout(output)
        output = self.ann1(x+output)

        x = output.clone()
        output = self.ff(inputs)
        output = self.dropout(output)
        output = self.ann2(x+output)

        return output

    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        '''
            MultiHeadAttention: 1
            Masked MultiHeadAttention: 1
            Add & LayerNorm: 3
            FeedForward: 1 
        '''
        self.batch_size = config.batch_size
        self.max_len=config.max_len
        self.model_dim = config.model_dim
        self.p_drop = config.p_drop

        self.dropout = nn.Dropout(self.p_drop)

        self.masked_mha = MultiHeadAttention()
        self.ann1 = nn.LayerNorm([self.batch_size, self.max_len, self.model_dim])

        self.mha = MultiHeadAttention()
        self.ann2 = nn.LayerNorm([self.batch_size, self.max_len, self.model_dim])

        self.ff = FeedForward()
        self.ann3 = nn.LayerNorm([self.batch_size, self.max_len, self.model_dim])
        
    def forward(self, inputs, encoded_output):
        '''
            Masked MultiHeadAttention --> Add&LayerNorm --> MultiHeadAttention --> Add&LayerNorm --> FeedForward --> Add&LayerNorm
        '''
        x = inputs.clone()
        output = self.masked_mha(inputs, masking=True, encoded_output=None)
        output = self.dropout(output)
        output = self.ann1(x+output)

        x = output.clone()
        output = self.mha(inputs, masking=False, encoded_output=encoded_output)
        output = self.dropout(output)
        output = self.ann2(x+output)

        x = output.clone()
        output = self.ff(inputs)
        output = self.dropout(output)
        output = self.ann3(x+output)

        return output