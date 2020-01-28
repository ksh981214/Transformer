import torch
class config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_len = 40
    lower_case = True

    #model
    model_dim = 512
    batch_size = 32
    num_heads = 8
    dim_K = 64 #dim_model / multi_head
    dim_V = 64 #dim_model / multi_head
    dim_ff = 2048 
    p_drop = 0.1
    N = 6

    #train
    num_epoch = 100
    beta1=0.9
    beta2=0.98
    eps=10e-9
    warmup_steps=4000
    step_num = 1

    eps_ls=0.1
    scheduler = False
    if scheduler:
        initial_lr = 1
    else:
        initial_lr = 0.0001

    train_set = 0.9 #test_set=0.1
    
    use_save_file = False 
    want_save_file = True
    use_file_len = batch_size * 10000 