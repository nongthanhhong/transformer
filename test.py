import torch
import math
from model import *


# e = nn.Embedding(4,3)
# a = torch.randint(2,(5,3))
# print(a)
# print(e(a))
# a = torch.rand(5, 5)
# sm = torch.nn.Softmax(dim =1)
# sm_a = sm(a)
# print(sm_a)

# v = torch.rand(5, 7)

# print(v)
# dot_v = torch.einsum("ij, jd -> id", a, v)
# print(dot_v)

# print(a+v)


# eb = torch.rand(5, 5)

# pe = torch.rand(5, 5)

# print(eb+pe)

import sys
from preprocess_data import *
torch.manual_seed(999)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

input_file = 'English-Vietnamese translation/en_test.txt'
output_file = 'English-Vietnamese translation/vi_test.txt'

train_data_loader, val_data_loader, input_tokenizer, output_tokenizer = Data(input_file, 
                                                                            output_file, 
                                                                            batch_size)
input_vocab_size = input_tokenizer.vocab_size()
output_vocab_size = output_tokenizer.vocab_size()

print(input_vocab_size, output_vocab_size)

model = Transformer(max_len=max_len_input, 
                    input_vocab_size= input_vocab_size,
                    output_vocab_size=output_vocab_size, 
                    num_layers=2,
                    heads=4,
                    d_model=8,
                    d_ff=7,
                    dropout=0.1,
                    bias=True).to(device)
model.train()

# batch_size = 5
# vocab = torch.arange(0, 9, 1, dtype=int).to(device)
x = torch.randint(10,(batch_size, max_len_input))
x_mask = torch.randint(2,(batch_size, 1, 1, max_len_input))
x_target = torch.randint(10,(batch_size, max_len_input))
target_mask = x_mask & (torch.triu(torch.ones((batch_size, 1, max_len_input, max_len_input)), diagonal=1).type(torch.int) == 0)

for i in (x, x_target, x_mask, target_mask):
    print(i.shape, ' = ', i)


# softmax_output_0 = model(x=x, x_mask=x_mask, x_target=x_target, target_mask=target_mask)
# print(softmax_output_0[0])

for batch in train_data_loader:
    x, output, x_target, x_mask, target_mask, _, _ = batch.values()
    for k, v in batch.items():
        try:
            print(k, v.shape, " = ", v)
        except:
            print(k, len(v), " = ", v)
    break

x = x.to(device) 
output = output.to(device) 
x_target = x_target.to(device) 
x_mask = x_mask.to(device) 
target_mask = target_mask.to(device) 

embed_input = model.input_embedding(x)
try:
    embed_output = model.output_embedding(output)
except:
    embed_output = model.output_embedding(x_target)

embed_w_pos_input = model.positional_encoding(embed_input)
embed_w_pos_output = model.positional_encoding(embed_output)

encode_input = model.encoder(x=embed_w_pos_input, x_mask=x_mask)

decoder_output = model.decoder(encode_x=encode_input, 
                                      x_mask=x_mask, 
                                      target_x=embed_w_pos_output, 
                                      target_mask=target_mask)

softmax_output_1 = model.generator(x=decoder_output)
print(softmax_output_1[0])

# from underthesea import text_normalize, word_tokenize
# import re 
# input_file = "English-Vietnamese translation\\vi_sentences.txt"

# with open(input_file, 'r', encoding='utf-8') as f:
#     input_data = f.readlines()

# input_data=input_data[:10]
# data = []
# for line in input_data:
#     line = re.sub(r'\W+$', '', line)
#     data.append(word_tokenize(text_normalize(line), format="text"))

# for i in data:
#     print(i)

# from preprocess_data import *
# import numpy as np
# import time
# # Set the file paths and other parameters
# input_file = 'English-Vietnamese translation/en_test.txt'
# output_file = 'English-Vietnamese translation/vi_test.txt'

# batch_size = 5

# # Create the data loader
# start_time = time.time()
# train_data_loader, val_data_loader= create_data_loader(input_file, 
#                                 output_file,
#                                 batch_size)

# print(f'data loader take: {time.time() - start_time} s')


# for batch in train_data_loader:
#     for v in batch.values():
#         try:
#             print(v.shape)
#         except:
#             print(len(v))

#     for k, v in batch.items():
#         try:
#             print(k, v.shape)
#         except:
#             print(k, len(v))
#     break
            

