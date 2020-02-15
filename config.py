import torch
class config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Preprocess
    batch_size = 32
    lower_case = True
    sentence_sorting= True
    use_save_file = False
    want_save_file = True
    use_file_len = batch_size * 200
    train_set = 0.9                     #test_set=0.1
    delete_thres = 0

    #Model
    model_dim = 512
    num_heads = 8
    dim_K = 64                          #dim_model / multi_head
    dim_V = 64                          #dim_model / multi_head
    dim_ff = 2048 
    p_drop = 0.1
    N = 6

    #Train
    num_epoch = 100
    beta1=0.9
    beta2=0.98
    eps=10e-9
    initial_lr = 1e-5
    use_lr_scheduler = True
    if use_lr_scheduler:
        warmup_steps=4000
        scheduler_scaling = 1e+3
    '''
        for using 2 * 32 * 10000,
        If loading consume 10 sec...
        else about 10 hours...
    '''
    label_smoothing = False
    if label_smoothing:
        eps_ls=0.1  

    #Test
    translate_max_len = 50                  