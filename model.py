import torch
import torch.nn as nn
import math

from config import config
from layer import Encoder,Decoder

config = config()

    
class Transformer(nn.Module):
    def __init__(self, src_word2idx, trg_word2idx):
        super().__init__()

        self.batch_size = config.batch_size
        self.model_dim = config.model_dim
        self.N = config.N
        self.dim_K = config.dim_K
        self.num_heads = config.num_heads
        self.device=config.device
            
        self.src_word2idx = src_word2idx
        self.trg_word2idx = trg_word2idx

        self.model_dim = config.model_dim

        self.src_embedding = nn.Embedding(len(self.src_word2idx),self.model_dim)
        self.trg_embedding = nn.Embedding(len(self.trg_word2idx),self.model_dim)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.linear = nn.Linear(self.dim_K * self.num_heads, len(self.trg_word2idx))
        self.softmax = torch.nn.Softmax(dim=2)
        
    def get_embedded_batch(self, sen_batch, embedding):
        '''
            sen_batch: Tensor, sentence by idx, batch_size x max_len
            output: Tensor, batch_size x max_len x model_dim
        '''
        embedded_batch=embedding(sen_batch).to(self.device)
        return embedded_batch

    def get_positional_encoding(self, max_len, model_dim):
        '''
            PE : max_len x model_dim , it is learned
        '''
        PE = torch.zeros(max_len, model_dim, dtype=torch.float32).to(self.device)
        for pos in range(max_len): 
            for i in range(0, model_dim, 2):
                PE[pos,i] = math.sin(pos/(10000 ** ((2*i)/model_dim)))
                PE[pos,i+1] = math.cos(pos/(10000 ** ((2*(i+1))/model_dim)))
        return PE

    def add_positional_encoding(self, sen, max_len, model_dim):
        '''
            sen : batch_size x max_len x model_dim
            PE  : max_len x model_dim
        '''
        PE = torch.zeros(sen.size()[1], model_dim, dtype=torch.float32).to(self.device)

        # Positional Encoding Initialization
        for pos in range(PE.size()[0]):                                         # max_len
            for i in range(0, model_dim, 2):                                 
                PE[pos,i] = math.sin(pos/(10000 ** ((2*i)/model_dim)))
                PE[pos,i+1] = math.cos(pos/(10000 ** ((2*(i+1))/model_dim)))

        # Add Positional Encoding
        for b in range(sen.size()[0]):                                          # batch_size
            sen[b] = sen[b]+PE
        
        return sen

    def forward(self, src, trg):
        '''
            Get Embedded Batch --> Add Positional Encoding --> Encoder 
            Get Embedded Batch --> Add Positional Encoding --> Decoder --> Linear --> Softmax --> Output Probabilities
            src : Tensor, batch_size x max_len
            trg : Tensor, batch_size x max_len
        '''
        src = self.get_embedded_batch(src, self.src_embedding) # batch_size x max_len x model_dim
        trg = self.get_embedded_batch(trg, self.trg_embedding) # batch_size x max_len x model_dim

        #Add Positional Encoding
        src = self.add_positional_encoding(src, src.size()[1], src.size()[2])
        trg = self.add_positional_encoding(trg, trg.size()[1], trg.size()[2])
    
        #Encoder
        encoded_output = []
        for i in range(self.N):
            src = self.encoder(src) #Encoded Output
            encoded_output.append(src)

        #Decoder
        for i in range(self.N):
            trg = self.decoder(trg, src)

        output = self.linear(trg)
        output = self.softmax(output)

        return output