from config import config

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

def train(p, transformer, print_term=1):
    '''
        p: preprocess object
        transformer: transformer object
        print_term: print loss's mean 
    '''
    '''
        label smoothing with Cross Entropy
        Out of Memory 

        Random Batch
        Ignore Index?
        Control Learning Rate
    '''
    def restore_sentence(idxs, idx2word):
        '''
            idxs: sentence by idxs
            return: sentence by word
        '''
        sen = []
        for idx in idxs:
            word = idx2word[idx.item()]
            #print(word)
            if word =='<BNK>':
                break
            else:
                sen.append(word)
        return sen
    device = config.device
    batch_size = config.batch_size
    num_epoch = config.num_epoch
    model_dim = config.model_dim
    warmup_steps = config.warmup_steps
    scheduler = config.scheduler

    #train mode, the difference exists at Dropout or BatchNormalization
    transformer.train()

    #Xavier Initialization
    for param in transformer.parameters():
        if param.dim()>1:
            nn.init.xavier_uniform_(param)
    
    initial_lr = config.initial_lr
    beta1=config.beta1
    beta2=config.beta2
    eps = config.eps

    opt = torch.optim.Adam(transformer.parameters(), lr=initial_lr, betas=(beta1, beta2), eps=eps)

    if scheduler:
        step_num = 0
        f = lambda step_num:model_dim**(-0.5) * torch.min(torch.tensor([(step_num+1)**(-0.5), (step_num+1)*warmup_steps**(-1.5)]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=f)

    #Start Epoch
    total_loss = 0
    
    #train
    output_prob = []
    start = time.time()
    initial_time = start
    temp = start

    train_src_idx = p.train_src_idx
    train_trg_idx = p.train_trg_idx
    for epoch in range(num_epoch):
        print("{} epoch".format(epoch+1))

        #Make Random Batch
        rands = np.arange(len(train_src_idx))
        np.random.shuffle(rands)

        src = train_src_idx[rands]
        trg = train_trg_idx[rands]
        src_batch = []
        trg_batch = []

        for i in range(int(len(src)/batch_size)):
            src_batch.append(src[i*batch_size:(i+1)*batch_size])
            trg_batch.append(trg[i*batch_size:(i+1)*batch_size])

        for i in range(len(src_batch)):
            #print(len(src_batch))
            if scheduler:
                step_num = step_num + 1
                scheduler.step(step_num)
                print('lr={}'.format(scheduler.get_lr()))

            opt.zero_grad()
            preds = transformer(src_batch[i], trg_batch[i])

            loss = F.cross_entropy(preds.view(-1,preds.size(-1)), trg_batch[i].view(-1).to(device))

            loss.backward()
            opt.step()

            total_loss = total_loss + loss

            if (i+1) % print_term == 0:
                loss_avg = total_loss / print_term
                #print("Total Consume time is {}".format(time.time()-initial_time))
                #print("{} batch Consume time is {}".format(print_term, time.time()-temp))
                print("loss avg is {}".format(loss_avg))
                total_loss = 0
                temp = time.time()
