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

    train(p, transformer)
    
    #test(p,transformer)
    
main()