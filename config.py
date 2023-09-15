import torch
import torch.nn as nn
from Sophia.sophia import SophiaG
import os


# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3

# Parameters for Transformer & training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

train_device = 'gpu' # or 'tpu'

ckpt_dir = 'ckpt/gpu_train'
os.makedirs(ckpt_dir, exist_ok=True)
saved_checkpoint_path = "ckpt/gpu_train/transformer_AdamW_epoch_7_loss_9.1570_BLEU_0.0_m9_d9_3h_54m.pt"

      # ckpt_dir = 'ckpt/tpu_train'
      # os.makedirs("ckpt_dir", exist_ok=True)
      # saved_checkpoint_path = "ckpt/tpu_train/transformer_AdamW_epoch_7_loss_9.1570_BLEU_0.0_m9_d9_3h_54m.pt"

learning_rate = 1e-4
batch_size = 32
max_len_input = 50
num_heads = 6
num_layers = 4
d_model = 300
d_ff = 2048
d_k = d_model // num_heads
drop_out_rate = 0.1
num_epochs = 10
beam_size = 8
log_interval = 500

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)

optimizer_name = "AdamW"

def create_optimizer(model_parameters):
 if optimizer_name == "AdamW":
    return torch.optim.AdamW(model_parameters, lr=learning_rate, betas=(0.9, 0.98),eps=1e-9)
 elif optimizer_name == "Sophia":
    return SophiaG(model_parameters, lr=learning_rate, rho=0.03)