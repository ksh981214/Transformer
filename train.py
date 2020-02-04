from config import config

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import copy

def train(p, transformer, print_term= 20):
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

    device = config.device
    batch_size = config.batch_size
    num_epoch = config.num_epoch
    model_dim = config.model_dim
    label_smoothing = config.label_smoothing
    if label_smoothing:
        eps_ls = config.eps_ls

    def put_padding(batch, word2ind):
        '''
            batch   :   * x batch_size x length x model_dim
        '''
        for i in range(len(batch)):
            # get max_len
            max_len = 0
            for sen in batch[i]:
                if len(sen)>max_len:
                    max_len = len(sen)
                else:
                    pass
            
            # put padding
            for j in range(len(batch[i])):
                sen = batch[i][j]
                if len(sen) < max_len:
                    while len(sen)!=max_len:
                        sen.append(word2ind['<BNK>'])
            # convert to tensor
            batch[i] = torch.tensor(batch[i]).to(device)

        return batch

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

    if label_smoothing:
        def make_label_smoothing(sen, class_num, alpha):
            '''
                a: hypyer_parameter(ex: 0.1, 0.9..)
                new_onehot_labels = (1-a) * y_hot + a/K 
            '''
            #print("before:{}".format(sen))

            smoothing_sen = list((1-alpha)*np.array(sen) + alpha/class_num)
            #print("after:{}".format(sen))
            return smoothing_sen

    scheduler = config.scheduler
    if scheduler:
        warmup_steps = config.warmup_steps

    #train mode, the difference exists at Dropout or BatchNormalization
    transformer.train()

    #Xavier Initialization
    # for param in transformer.parameters():
    #     if param.dim()>1:
    #         nn.init.xavier_uniform_(param)
    
    initial_lr = config.initial_lr
    beta1=config.beta1
    beta2=config.beta2
    eps = config.eps

    opt = torch.optim.Adam(transformer.parameters(), lr=initial_lr, betas=(beta1, beta2), eps=eps)

    if scheduler:
        step_num = 1
        f = lambda step_num:(1e+4)*model_dim**(-0.5) * torch.min(torch.tensor([(step_num+1)**(-0.5), (step_num+1)*warmup_steps**(-1.5)]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=f)

    #Start Epoch
    total_loss = 0
    
    #train
    output_prob = []
    start = time.time()
    initial_time = start
    temp = start

    src = p.train_src_idx     #list
    trg = p.train_trg_idx     #list
    
    #Label_Smoothing
    if label_smoothing:
        smoothing_trg = copy.deepcopy(trg)
        print("Start Making Label Smoothing")
        start = time.time()
        for i in range(len(smoothing_trg)):
            smoothing_trg[i] = make_label_smoothing(smoothing_trg[i], len(p.trg_word2ind), eps_ls)
        print("Finish Making Label Smoothing")
        print("Consume Time: {}".format(time.time()-start))

    src_batch = []
    trg_batch = []
    if label_smoothing:
        smoothing_trg_batch =[]

    for i in range(int(len(src)/batch_size)):
        src_batch.append(src[i*batch_size:(i+1)*batch_size])
        trg_batch.append(trg[i*batch_size:(i+1)*batch_size])
        if label_smoothing:
            smoothing_trg_batch.append(smoothing_trg[i*batch_size:(i+1)*batch_size])

    src_batch = put_padding(src_batch, p.src_word2ind)
    trg_batch = put_padding(trg_batch, p.trg_word2ind)
    if label_smoothing:
        smoothing_trg_batch = put_padding(smoothing_trg_batch, p.trg_word2ind)

    for epoch in range(num_epoch):
        epoch_start = time.time()
        print("{} epoch".format(epoch+1))
        epoch_loss = 0
        for i in range(len(src_batch)):
            #print(len(src_batch))

            opt.zero_grad()

            preds = transformer(src_batch[i], trg_batch[i])

            if label_smoothing:
                loss = F.cross_entropy(preds.view(-1,preds.size(-1)), smoothing_trg_batch[i].view(-1), ignore_index=transformer.src_word2ind['<BNK>'])
            else:
                loss = F.cross_entropy(preds.view(-1,preds.size(-1)), trg_batch[i].view(-1))

            loss.backward()
            opt.step()
            if scheduler:
                scheduler.step()
                
            total_loss = total_loss + loss
            epoch_loss = epoch_loss + loss
            if (i+1) % print_term == 0:
                loss_avg = total_loss / print_term
                print("{} Batch Loss Avg is {}".format(print_term, loss_avg))
                print("Learning Rate={}".format(opt.param_groups[0]['lr']))
                total_loss = 0
                print(src_batch[i][0])
                print()
                print(trg_batch[i][0])
                print()
                print(preds[0])
                print(torch.argmax(preds[0], dim=1))
                print("------------------------------")
        # print Epoch loss
        print("Epoch Loss Avg is {}".format(epoch_loss/len(src_batch)))
        print("Consume time per this epoch is {}".format(time.time()-epoch_start))
        epoch_loss = 0