import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import math
from model import *
import sys
from preprocess_data import *
from nltk.translate import bleu, bleu_score

torch.manual_seed(999)
input_file = 'English-Vietnamese translation/en_test.txt'
output_file = 'English-Vietnamese translation/vi_test.txt'

train_data_loader, val_data_loader, input_tokenizer, output_tokenizer = Data(input_file, 
                                                                            output_file, 
                                                                            1,
                                                                            eval = True)
input_vocab_size = input_tokenizer.vocab_size()
output_vocab_size = output_tokenizer.vocab_size()

print('vocab size of input and output:', input_vocab_size, output_vocab_size)
print('size of train and val: ',len(train_data_loader), len(val_data_loader))

model = Transformer(max_len=max_len_input,
                        input_vocab_size=input_vocab_size,
                        output_vocab_size=output_vocab_size,
                        num_layers=num_layers,   
                        heads=num_heads, 
                        d_model=d_model, 
                        d_ff=d_ff, 
                        dropout=drop_out_rate, 
                        bias=True).to(device)

model.eval()
checkpoint_path = "ckpt/transformer_AdamW_epoch_6_loss_9.2316_BLEU_0.0_m9_d8_9h_42m.pt"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# print('check shape data loader')
total_bleu = 0
for batch in train_data_loader:
    x, output, x_target, x_mask, target_mask, in_text, out_text = batch.values()
    x = x.to(device)
    output = output.to(device) 
    x_target = x_target.to(device) 
    x_mask = x_mask.to(device) 
    target_mask = target_mask.to(device)

    # for k, v in batch.items():
    #     try:
    #         print(k, v.shape, " = ", v)
    #     except:
    #         print(k, len(v), " = ", v)
    
    embed_input = model.input_embedding(x)
    embed_output = model.output_embedding(output)

    embed_w_pos_input = model.positional_encoding(embed_input)
    embed_w_pos_output = model.positional_encoding(embed_output)

    encode_input = model.encoder(x=embed_w_pos_input, x_mask=x_mask)
    decoder_output = model.decoder(encode_x=encode_input, 
                                        x_mask=x_mask, 
                                        target_x=embed_w_pos_output, 
                                        target_mask=target_mask)

    softmax_output_1 = model.generator(x=decoder_output)
    
    predict_index = torch.argmax(softmax_output_1, dim=-1)
    print(predict_index.shape, x_target.shape)
    print('input: ', x)
    print('input mask: ', x_mask)
    print('output: ', output)
    print('output mask: ', target_mask)
    print('target: ', x_target)
    print('predict: ', predict_index)

    total_bleu += bleu(predict_index, x_target)
    break

print('BLEU score: ',total_bleu/len(train_data_loader))