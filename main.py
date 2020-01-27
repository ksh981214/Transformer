from preprocess import Preprocess
from model import Transformer
from train import train

from config import config

config = config()

def main():
    device = config.device
    
    
    p = Preprocess("data/europarl-v7.fr-en.en", "data/europarl-v7.fr-en.fr")
    
    transformer = Transformer(p.src_word2ind, p.trg_word2ind)
    transformer.to(device)
#     #len(p.src_idx)
#     src_batch = []
#     trg_batch = []

#     for i in range(int(len(p.src_idx)/config.batch_size)):
#         src_batch.append(p.src_idx[i*config.batch_size:(i+1)*config.batch_size])
#         trg_batch.append(p.trg_idx[i*config.batch_size:(i+1)*config.batch_size])

#     print("src batch # is {}".format(len(src_batch)))
#     print("trg batch # is {}".format(len(trg_batch)))
    
#     print(transformer(src_batch[0], trg_batch[0])[0][0])
    train(p, transformer)
    
main()