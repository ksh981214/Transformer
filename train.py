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
            if word =='<EOS>':
                sen.append(word)
                
            else:
                sen.append(word)
        return sen

    if label_smoothing:
        def make_label_smoothing(sen, class_num, alpha):
            '''
                a: hypyer_parameter(ex: 0.1, 0.9..)
                new_onehot_labels = (1-a) * y_hot + a/K 
            '''
            smoothing_sen = list((1-alpha)*np.array(sen) + alpha/class_num)
            return smoothing_sen

    use_lr_scheduler = config.use_lr_scheduler
    if use_lr_scheduler:
        warmup_steps = config.warmup_steps
        scheduler_scaling = config.scheduler_scaling

    #train mode, the difference exists at Dropout or BatchNormalization
    transformer.train()
    
    initial_lr = config.initial_lr
    beta1=config.beta1
    beta2=config.beta2
    eps = config.eps

    opt = torch.optim.Adam(transformer.parameters(), lr=initial_lr, betas=(beta1, beta2), eps=eps)
    if use_lr_scheduler:
        lambda_f = lambda step_num:scheduler_scaling*model_dim**(-0.5) * torch.min(torch.tensor([(step_num+1)**(-0.5), (step_num+1)*warmup_steps**(-1.5)]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda_f)

    #Start Epoch
    total_loss = 0
    
    #train
    output_prob = []
    start = time.time()
    initial_time = start
    temp = start

    src = p.train_src_idx     #list
    trg = p.train_trg_idx     #list
    #nosos_trg = p.train_nosos_trg_idx     #list
    
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
    #nosos_trg_batch=[]

    if label_smoothing:
        smoothing_trg_batch =[]

    for i in range(int(len(src)/batch_size)):
        src_batch.append(src[i*batch_size:(i+1)*batch_size])
        trg_batch.append(trg[i*batch_size:(i+1)*batch_size])
        #nosos_trg_batch.append(nosos_trg[i*batch_size:(i+1)*batch_size])

        if label_smoothing:
            smoothing_trg_batch.append(smoothing_trg[i*batch_size:(i+1)*batch_size])

    src_batch = put_padding(src_batch, p.src_word2ind)
    trg_batch = put_padding(trg_batch, p.trg_word2ind)
    #nosos_trg_batch = put_padding(nosos_trg_batch, p.trg_word2ind)
    if label_smoothing:
        smoothing_trg_batch = put_padding(smoothing_trg_batch, p.trg_word2ind)

    for epoch in range(num_epoch):
        epoch_start = time.time()
        epoch_loss = 0
        for i in range(len(src_batch)):
            opt.zero_grad()

            #Shifted Right
            sos_tokens = torch.ones(trg_batch[i].size()[0],1).long().to(device) * p.trg_word2ind['<SOS>']        
            model_trg_input = torch.cat((sos_tokens, trg_batch[i]), dim=-1)
            model_trg_input = model_trg_input[:,:-1]

            preds = transformer(src_batch[i], model_trg_input)

            if label_smoothing:
                loss = F.cross_entropy(preds.view(-1,preds.size(-1)), smoothing_trg_batch[i].view(-1), ignore_index=p.trg_word2ind['<BNK>'])
            else:
                loss = F.cross_entropy(preds.view(-1,preds.size(-1)), trg_batch[i].view(-1), ignore_index=p.trg_word2ind['<BNK>'])        #No <SOS> because Shifted Right

            loss.backward()
            
            opt.step()
            if use_lr_scheduler:
                scheduler.step()

            total_loss = total_loss + loss
            epoch_loss = epoch_loss + loss
            if (i+1) % print_term == 1:
                #pdb.set_trace()
                loss_avg = total_loss / print_term
                print("{} epoch | {} Batch Loss Avg : {} | Learning Rate : {}".format(epoch+1, print_term, loss_avg, opt.param_groups[0]['lr']))
                total_loss = 0
                print(restore_sentence(src_batch[i][0], p.src_ind2word))
                print()
                print(preds[0])
                print()
                print(restore_sentence(trg_batch[i][0,:-1], p.trg_ind2word))
                #print(restore_sentence(nosos_trg_batch[i][0], p.trg_ind2word))
                print(restore_sentence(torch.argmax(preds[0], dim=1), p.trg_ind2word))
                print("---------------------------------------------------------------------------------------------------")
        # print Epoch loss
        print("Epoch Loss Avg : {} | Consume time per this epoch : {}".format(epoch_loss/len(src_batch), time.time()-epoch_start))
        epoch_loss = 0