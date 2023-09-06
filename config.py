import torch
import torch.nn as nn


# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3

# Parameters for Transformer & training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 1e-4
batch_size = 1
max_len_input = 10
num_heads = 4
num_layers = 2
d_model = 8
d_ff = 8
d_k = d_model // num_heads
drop_out_rate = 0.1
num_epochs = 3
beam_size = 8
ckpt_dir = 'saved_model'
log_interval = 100

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss(ignore_index=0).to(device)

def optimizer(model_parameters):
 return torch.optim.Adam(model_parameters, lr=learning_rate, eps=1e-9)