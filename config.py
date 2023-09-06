import torch
import torch.nn as nn
from Sophia.sophia import SophiaG


# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3

# Parameters for Transformer & training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
learning_rate = 1e-4
batch_size = 32
max_len_input = 100
num_heads = 4
num_layers = 6
d_model = 256
d_ff = 256
d_k = d_model // num_heads
drop_out_rate = 0.1
num_epochs = 10
beam_size = 8
ckpt_dir = 'ckpt'
log_interval = 100

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)

optimizer_name = "AdamW"
def create_optimizer(model_parameters):
 if optimizer_name == "AdamW":
    return torch.optim.AdamW(model_parameters, lr=learning_rate, eps=1e-9)
 else:
    return SophiaG(model_parameters, lr=learning_rate, rho=0.03)