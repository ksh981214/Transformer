'''
    That does not work. I think the reason why that(shifted right) does not work is 'The difference of Size of Data..'
    I will make test code later..
'''
# import torch 

# def test(p, transformer):
#     '''
#         p: preprocess object
#         transformer: transformer object
#     '''

#     def restore_sentence(idxs, idx2word):
#         '''
#             idxs: sentence by idxs
#             return: sentence by word
#         '''
#         sen = []
#         for idx in idxs:
#             word = idx2word[idx.item()]
#             #print(word)
#             if word =='<BNK>':
#                 break
#             else:
#                 sen.append(word)
#         return sen
#     #evaluation mode, the difference exists at Dropout or BatchNormalization
#     transformer.eval()

#     src = p.test_src_idx
#     trg = p.test_trg_idx

#     for i in range(len(src)):
#         preds = transformer(src.unsqueeze(0), trg.unsqueeze(0))
#         preds = preds.squeeze()
#         preds = torch.argmax(preds, dim=1)
        
#         preds_sentence = restore_sentence(preds, p.trg_ind2word)
    
