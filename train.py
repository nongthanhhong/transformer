import torch.nn as nn
import torch
import time
import tqdm as tqdm
import sys, os
import numpy as np
import logging
import argparse
from model import Transformer
import datetime
from torch.utils.tensorboard import SummaryWriter

from config import *
from preprocess_data import *
from nltk.translate import bleu, bleu_score


# Define the train function
def train(train_data_loader, model, loss_fn, optimizer, scheduler, writer, log_interval):

    # Set the model to training mode
    model.train()
    total_loss = 0
    batch_total_loss = 0
    for i, batch in enumerate(train_data_loader):
        torch.cuda.empty_cache()
        input, output, output_target, input_mask, output_mask, _, _ = batch.values()

        input = input.to(device)
        output = output.to(device)
        output_target = output_target.to(device)
        input_mask = input_mask.to(device)
        output_mask = output_mask.to(device)

        # Zero out gradients from previous iteration
        optimizer.zero_grad()

        # Pass source and target data to model to obtain output logits.
        predict = model(x=input,
                        x_mask=input_mask,
                        x_target=output,
                        target_mask=output_mask)

        # Compute the loss between the output and target data
        loss = loss_fn(predict.view(-1, predict.shape[-1]), output_target.view(-1))
        # loss = loss_fn(torch.argmax(predict, dim=2), output_target.view(-1))

        # Compute gradients
        loss.backward()

        # do some tricks
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update model parameters
        optimizer.step()
        
        # Update the learning rate
        scheduler.step()

        # Accumulate total loss
        total_loss += loss.item()
        batch_total_loss += loss.item()

        # log during train
        if (i+1) % log_interval == 0:
            avg_loss = batch_total_loss / log_interval
            print(f'\tBatch: {i+1}/{len(train_data_loader)} | Loss: {avg_loss:.4f}')
            writer.add_scalar("Loss/train_batch", avg_loss, i + 1)
            batch_total_loss = 0

    # Return average training loss
    return total_loss / len(train_data_loader)

# Define the save_checkpoint and load_checkpoint functions
def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path):
    # Save the state of the model, optimizer, and current epoch to the specified checkpoint file
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }, checkpoint_path)
    print('\tSaved checkpoint!')

def load_checkpoint(model, optimizer, checkpoint_path):
    # Load the state of the model, optimizer, and current epoch from the specified checkpoint file
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # Return the current epoch
    return epoch, model, optimizer

# Define the log_progress function
def log_progress(epoch, train_loss, val_loss=None, avg_bleu = None):
    # Log the current training epoch and training loss
    print(f'\tEPOCH: {epoch+1} | Train Loss: {train_loss:.4f}', end='')
    if val_loss is not None:
        # If validation loss is provided, log it as well
        print(f' | Val Loss: {val_loss:.4f}', end='')
    print(f' | BLEU score: {avg_bleu:.2f}', end='')

# Define the evaluate function
def evaluate(val_data_loader, model, loss_fn):
    # Set the model to evaluation mode
    model.eval()
    total_loss = 0
    total_bleu = 0
    with torch.no_grad():
        for batch in val_data_loader:
            input, output, output_target, input_mask, output_mask, _, _ = batch.values()

            input = input.to(device)
            output = output.to(device)
            output_target = output_target.to(device)
            input_mask = input_mask.to(device)
            output_mask = output_mask.to(device)

            # Pass source and target data to model to obtain output logits.
            predict = model(x=input,
                        x_mask=input_mask,
                        x_target=output,
                        target_mask=output_mask)

            # Compute the loss between the output and target data.
            loss = loss_fn(predict.view(-1, predict.shape[-1]), output_target.view(-1))
            # Accumulate total loss.
            total_loss += loss.item()
            predict_index = torch.argmax(predict, dim=-1)
            total_bleu += bleu(predict_index, output_target)

    # Return average validation loss.
    avg_bleu = total_bleu / len(val_data_loader)
    return total_loss / len(val_data_loader), avg_bleu

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config-path', type=str, required=True, help="path to config file for train model")

    # args = parser.parse_args()

    
    # Set the file paths and other parameters
    input_file = 'English-Vietnamese translation/en_sentences.txt'
    output_file = 'English-Vietnamese translation/vi_sentences.txt'

    train_data_loader, val_data_loader, input_tokenizer, output_tokenizer = Data(input_file, 
                                                                                 output_file, 
                                                                                 batch_size)
    input_vocab_size = input_tokenizer.vocab_size()
    output_vocab_size = output_tokenizer.vocab_size()

    print(f'Size of English vocab: {input_vocab_size}\nSize of Vietnamese vocab: {output_vocab_size}')

    # Create the model
    model = Transformer(max_len=max_len_input,
                        input_vocab_size=input_vocab_size,
                        output_vocab_size=output_vocab_size,
                        num_layers=num_layers,   
                        heads=num_heads, 
                        d_model=d_model, 
                        d_ff=d_ff, 
                        dropout=drop_out_rate, 
                        bias=True).to(device)

    # Define your optimizer
    optimizer = create_optimizer(model.parameters())

    checkpoint_path = "ckpt/transformer_AdamW_epoch_7_loss_9.1570_BLEU_0.0_m9_d9_3h_54m.pt"
    _, model, optimizer = load_checkpoint(model, optimizer, checkpoint_path)
    # Create a SummaryWriter instance for TensorBoard logging
    writer = SummaryWriter()

    # setup warmup schedule and optimizer
    # Define total number of training steps
    total_steps = len(train_data_loader)
    # Define warmup proportion
    warmup_proportion = 0.1
    # Compute the number of warmup steps
    warmup_steps = int(total_steps * warmup_proportion)
    # Define your scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, 1))

    # Train the model
    print("\n=============== Start Training Phase ===============")
    start_time = time.time()
    best_epoch = 0
    pre_loss = 10**9
    for epoch in range(num_epochs):
        print(f'EPOCH {epoch + 1}:')
        train_loss = train(train_data_loader, 
                           model,
                           loss_fn,
                           optimizer,
                           scheduler,
                           writer,
                           log_interval)
        val_loss, avg_bleu = evaluate(val_data_loader,
                            model,
                            loss_fn)
        log_progress(epoch, train_loss, val_loss, avg_bleu)
        writer.add_scalar("Loss/train_epoch", train_loss, epoch, val_loss)
        
        #check best epoch
        if val_loss < pre_loss:
            pre_loss = val_loss
            best_epoch = epoch + 1
            
            now = datetime.datetime.now()
            checkpoint_path = f"{ckpt_dir}/transformer_{optimizer_name}_epoch_{epoch}_loss_{val_loss:.4f}_BLEU_{avg_bleu}_m{now.month}_d{now.day}_{now.hour}h_{now.minute}m.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        
    end_time = time.time() - start_time
    print(f'Training take: {round(end_time, 3)} seconds ~ {round(end_time/60, 3)} minutes ~ {round(end_time/3600, 3)} hours')
    print(f'Best Epoch: {best_epoch} with val_loss = {round(pre_loss, 8)}')

