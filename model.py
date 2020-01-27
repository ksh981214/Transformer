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
        self.max_len = config.max_len
        self.model_dim = config.model_dim
        self.N = config.N
        self.dim_K = config.dim_K
        self.num_heads = config.num_heads

        self.src_word2idx = src_word2idx
        self.trg_word2idx = trg_word2idx

        self.model_dim = config.model_dim

        self.src_embedding = nn.Embedding(len(self.src_word2idx),self.model_dim)
        self.trg_embedding = nn.Embedding(len(self.trg_word2idx),self.model_dim)

        self.encoder_pe = self.get_positional_encoding(self.max_len, self.model_dim) #max_len x model_dim
        self.decoder_pe = self.get_positional_encoding(self.max_len, self.model_dim)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.linear = nn.Linear(self.dim_K * self.num_heads, len(self.trg_word2idx))
        self.softmax = torch.nn.Softmax(dim=2)

    def get_embedded_batch(self, sen_batch, embedding):
        '''
            sen_batch: Tensor, sentence by idx, batch_size x max_len
            output: batch_size x max_len x model_dim
        '''
        #embedded_batch=embedding(torch.tensor(sen_batch, dtype= torch.long))
        embedded_batch=embedding(sen_batch)
        
        return embedded_batch

    def get_positional_encoding(self, max_len, model_dim):
        '''
            PE : max_len x model_dim , it is learned
        '''
        PE = torch.zeros(max_len, model_dim, dtype=torch.float32)
        for pos in range(max_len): 
            for i in range(0, model_dim, 2):
                PE[pos,i] = math.sin(pos/(10000 ** ((2*i)/model_dim)))
                PE[pos,i+1] = math.cos(pos/(10000 ** ((2*(i+1))/model_dim)))
        return PE

    def forward(self, src, trg):
        '''
            Get Embedded Batch --> Add Positional Encoding --> Encoder 
            Get Embedded Batch --> Add Positional Encoding --> Decoder --> Linear --> Softmax --> Output Probabilities

            src : Tensor, batch_size x max_len
            trg : Tensor, batch_size x max_len
        '''
        src = self.get_embedded_batch(src, self.src_embedding) # batch_size x max_len x model_dim
        trg = self.get_embedded_batch(trg, self.trg_embedding) # batch_size x max_len x model_dim
        #print("Get Eembedded Batch!")
        #print("src Size: {}".format(src.size()))
        #print("trg Size: {}".format(trg.size()))
        #Add Positional Encoding
        for i in range(self.batch_size):
            src[i] = src[i] + self.encoder_pe
            trg[i] = trg[i] + self.decoder_pe

        #Encoder
        encoded_output = []
        for i in range(self.N):
            src = self.encoder(src) #Encoded Output
            encoded_output.append(src)

        #Decoder
        for i in range(self.N):
            trg = self.decoder(trg, encoded_output[i])
        
        #print("trg size:",trg.size())
        output = self.linear(trg)
        #print("linear size:",output.size())
        output = self.softmax(output)
        #print(output.size())
        #print(output[0].size())
        ##print(output[0][0].size())
        #print("output's first row's sum is {}".format(torch.sum(output[0,0,:])))

        return output