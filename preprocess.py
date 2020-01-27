import nltk #Tokenizer 사용을 위해
from nltk.tokenize import TreebankWordTokenizer

import re #정규표현식 사용
import time
import torch

from collections import Counter
from config import config

config = config()


class Preprocess():
    def __init__(self, src_path, trg_path):
        self.max_len=config.max_len
        self.lower_case = config.lower_case
        

        start = time.time()

        src = open(src_path,'r',encoding='utf-8')
        trg = open(trg_path,'r',encoding='utf-8')
        
        print("Finish Data Loading")
        print("Consume Time: {}".format(time.time()-start))
        temp = time.time()
        
        src_lines=src.readlines() #2007723
        trg_lines=trg.readlines() #2007723

        #임시    
        src_lines = src_lines[:32*100]
        trg_lines = trg_lines[:32*100]

        print("Finish Data Reading")
        print("Data len: {}".format(len(src_lines)))
        print("Consume Time: {}".format(time.time()-temp))
        temp = time.time()

        #Tokenizing Sentence...
        tokenizer = TreebankWordTokenizer()
        r = re.compile("[A-Za-z0-9]") #특수문자 제거
        
        src_lines_tokenized, src_corpus = self.make_tokenized(src_lines, tokenizer, r)
        trg_lines_tokenized, trg_corpus = self.make_tokenized(trg_lines, tokenizer, r)

        src_frequency = Counter(src_corpus)
        trg_frequency = Counter(trg_corpus)

        src_processed = self.delte_rare_word(src_corpus, src_frequency, 2)
        trg_processed = self.delte_rare_word(trg_corpus, trg_frequency, 2)

        src_frequency = Counter(src_processed)
        trg_frequency = Counter(trg_processed)
        src_vocab = set(src_processed)
        trg_vocab = set(trg_processed)

        print("Finish Tokenizing Sentence")
        print("Consume Time: {}".format(time.time()-temp))
        temp = time.time()

        self.src_word2ind, self.src_ind2word = self.make_mapping(src_vocab, BNK=True, EOS=True, UNK=True, SOS=False)
        self.trg_word2ind, self.trg_ind2word = self.make_mapping(trg_vocab, BNK=True, EOS=True, UNK=True, SOS=True)

        print("src wor2ind # is {}".format(len(self.src_word2ind))) #5175(10000문장 기준)
        print("trg wor2ind # is {}".format(len(self.trg_word2ind))) #6547
        print("Finish Mapping Sentence")
        print("Consume Time: {}".format(time.time()-temp))
        temp = time.time()
        
        #Sentence by Word Idx
        src_idx = self.make_idx_sen(src_lines_tokenized, self.src_word2ind, BNK=True, EOS=True, UNK=True, SOS=False)
        trg_idx = self.make_idx_sen(trg_lines_tokenized, self.trg_word2ind, BNK=True, EOS=True, UNK=True, SOS=True)

        num_train_data = int(len(src_idx)*config.train_set)
    
        self.train_src_idx = torch.tensor(src_idx[:num_train_data]).to(config.device)
        self.train_trg_idx = torch.tensor(trg_idx[:num_train_data]).to(config.device)

        print("Train set len is {}".format(len(self.train_src_idx)))

        self.test_src_idx = torch.tensor(src_idx[num_train_data:]).to(config.device)
        self.test_trg_idx = torch.tensor(trg_idx[num_train_data:]).to(config.device)

        print("Test set len is {}".format(len(self.test_src_idx)))
        
        print("Finish Making Sentence By Idx")
        print("Consume Time: {}".format(time.time()-temp))

        print("Preprocess Finish!")
        print("Total Consume Time: {}".format(time.time()-start))

        
    def make_tokenized(self, lines_lst, tokenizer, r = re.compile("[A-Za-z0-9]")):
        '''
            lines_lst:  list of all lines
            tokenizer:  tokenizer obj
            r:          delete special character
        '''
        lines_tokenized = []
        corpus = []
        
        for i in range(len(lines_lst)):
            temp = tokenizer.tokenize(lines_lst[i])
            idx=0

            if self.lower_case:
                lower_temp= []

            if r is not None:
                for j in range(len(temp)):

                    if r.search(temp[idx]):
                        lower_temp.append(temp[idx].lower())
                        idx = idx+1
                    else:
                        temp.remove(temp[idx])
            #print(temp)
            if self.lower_case:
                lines_tokenized.append(lower_temp)
                corpus = corpus + lower_temp
            else:
                lines_tokenized.append(temp)
                corpus = corpus + temp

        return lines_tokenized, corpus

    def delte_rare_word(self, corpus, freq, cnt):
        '''
            corpus: list of all words(Allow Duplicate)
            freq:   frequency dictionary
            cnt:    thresold of delete
        '''
        processed = []
        for w in corpus:
            if freq[w]>2:
                processed.append(w)

        return processed

    def make_mapping(self, vocab, BNK, EOS, UNK, SOS):
        '''
            vocab: list of all words(Allow Duplicate)
        '''
        word2ind = {}
        i=0
        if BNK:
            word2ind['<BNK>']=i
            i=i+1
        if EOS:
            word2ind['<EOS>']=i
            i=i+1
        if UNK:
            word2ind['<UNK>']=i
            i=i+1
        if SOS:
            word2ind['<SOS>']=i
            i=i+1
        for sen in vocab:
            for word in sen.split():
                if word not in word2ind.keys():
                    word2ind[word] = i
                    i+=1
        
        ind2word = {}
        for k,v in word2ind.items():
            ind2word[v]=k
        
        return word2ind, ind2word

    def make_idx_sen(self, tokenized, word2ind, BNK, EOS, UNK, SOS):
        '''
            tokenized: list of all sentences , [[...],[...]]
        '''
        idx_sen = []
        for sen in tokenized:
            if SOS:
                new_sen=[word2ind['<SOS>']]
            else:
                new_sen=[]
            for word in sen:
                if word in word2ind.keys():
                    new_sen.append(word2ind[word])
                else:
                    new_sen.append(word2ind['<UNK>'])
            new_sen.append(word2ind['<EOS>'])
            if len(new_sen) < self.max_len:
                while len(new_sen)!=self.max_len:
                    new_sen.append(word2ind['<BNK>'])
            #if len > max_len, then cut..
            if len(new_sen) > self.max_len:
                new_sen = new_sen[:self.max_len]
            idx_sen.append(new_sen)
            
        return idx_sen