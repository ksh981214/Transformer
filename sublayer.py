import torch
import torch.nn as nn
import numpy as np

from config import config

config = config()

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_dim = config.model_dim
        self.dim_ff = config.dim_ff

        self.first_linear = nn.Linear(self.model_dim, self.dim_ff, bias=True) 
        self.second_linear = nn.Linear(self.dim_ff, self.model_dim, bias=True) 

        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        output = self.relu(self.first_linear(inputs))
        output = self.second_linear(output)
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_dim = config.model_dim
        self.num_heads = config.num_heads
        self.dim_K = config.dim_K
        self.dim_V = config.dim_V

        self.queries_linear= nn.Linear(self.dim_K * self.num_heads, self.model_dim)
        self.keys_linear= nn.Linear(self.dim_K * self.num_heads, self.model_dim)
        self.values_linear= nn.Linear(self.dim_V * self.num_heads, self.model_dim)

        self.output_linear = nn.Linear(self.dim_V * self.num_heads, self.model_dim)
    def attention(self,Q,K,V,masking):
        '''
            Q: Query,   batch_size x max_len x dim_K
            K: Key,     batch_size x max_len x dim_K
            V: Value,   batch_size x max_len x dim_V
        '''
        batch_size = Q.size()[0]
        max_len = Q.size()[1]
        dim_K = Q.size()[2]

        QKT = torch.matmul(Q,torch.transpose(K,1,2))            #max_len x max_len
        
        #scaling
        QKT = QKT / (dim_K ** 0.5)
        
        #masking
        if masking:
            '''
                trimat
                tensor([[0., 1.],
                        [0., 0.]])
            '''
            tri_mat = torch.tensor(np.triu(np.ones((batch_size, max_len, max_len)), k=1).astype('float32'))
            mask = torch.ones_like(tri_mat) * -1.0e9
            QKT = torch.where(torch.eq(tri_mat,0),QKT, mask)    #if tri_mat == 0 then QKT, else -1.0e9
            
        s = torch.nn.Softmax(dim=2)
        softmax = s(QKT)
        
        output = torch.matmul(softmax,V)    # batch_size x max_len x dim_V
        return output 

    def forward(self, inputs, masking=False, encoded_output=None):
        if encoded_output is not None:
            queries = inputs
            keys = encoded_output
            values = encoded_output
        else:
            queries = inputs
            keys = inputs
            values = inputs

        Q = self.queries_linear(queries)
        K = self.keys_linear(keys)
        V = self.values_linear(values)

        Q_split=torch.split(Q,int(self.model_dim/self.num_heads),dim=2) # 8 x (batch_size x max_len x model_dim/num_heads)
        K_split=torch.split(K,int(self.model_dim/self.num_heads),dim=2)
        V_split=torch.split(V,int(self.model_dim/self.num_heads),dim=2)

        #Scaled Dot_Product_Attention
        result_lst = []
        for (i,split) in enumerate(Q_split):
            result_lst.append(self.attention(Q_split[i],K_split[i],V_split[i],masking))


        #Concat
        concat = torch.cat(result_lst,dim=2) # 32 x 40 x 512
        output = self.output_linear(concat)

        return output