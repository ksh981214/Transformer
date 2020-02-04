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
        self.model_dim = config.model_dim
        self.p_drop = config.p_drop

        self.dropout = nn.Dropout(self.p_drop)

        self.mha = MultiHeadAttention()
        self.ann1 = nn.LayerNorm([self.model_dim])

        self.ff = FeedForward()
        self.ann2 = nn.LayerNorm([self.model_dim])

    def forward(self, inputs):
        '''
            MultiHeadAttention --> Add&LayerNorm --> FeedForward --> Add&LayerNorm
        '''
        x = inputs

        # MultiHeadAttention
        x = self.ann1(x+self.dropout(self.mha(x, masking=False, encoded_output=None)))

        # FeedForward
        x = self.ann2(x+self.dropout(self.ff(x)))

        return x

    
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
        self.model_dim = config.model_dim
        self.p_drop = config.p_drop

        self.dropout = nn.Dropout(self.p_drop)

        self.masked_mha = MultiHeadAttention()
        self.ann1 = nn.LayerNorm([self.model_dim])

        self.mha = MultiHeadAttention()
        self.ann2 = nn.LayerNorm([self.model_dim])

        self.ff = FeedForward()
        self.ann3 = nn.LayerNorm([self.model_dim])
    def forward(self, inputs, encoded_output):
        '''
            Masked MultiHeadAttention --> Add&LayerNorm --> MultiHeadAttention --> Add&LayerNorm --> FeedForward --> Add&LayerNorm
        '''

        x = inputs

        # Masked MultiHead Attention
        x = self.ann1(x+self.dropout(self.masked_mha(x, masking=True, encoded_output=None)))

        # MultiHead Attention
        x = self.ann2(x+self.dropout(self.mha(x, masking=False, encoded_output=encoded_output)))

        # FeedForward
        x = self.ann3(x+self.dropout(self.ff(x)))

        return x