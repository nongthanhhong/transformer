import torch.nn as nn
import torch
import time
import tqdm as tqdm
import sys, os
import numpy as np
import logging
import argparse
from model import Transformer
from torch.utils.tensorboard import SummaryWriter

from config import *
from preprocess_data import *


# Define the train function
def train(data_loader, model, loss_fn, optimizer, writer, log_interval):

    logging.info("\n=================Start Training Phase============")

    # Set the model to training mode
    # torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    for i, batch in enumerate(data_loader):

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
        # Compute gradients
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Accumulate total loss
        total_loss += loss.item()

        # log during train
        if (i + 1) % log_interval == 0:
            avg_loss = total_loss / log_interval
            print(f'Batch: {i+1}/{len(data_loader)} | Loss: {avg_loss:.4f}')
            writer.add_scalar("Loss/train_batch", avg_loss, i)
            total_loss = 0

    # Return average training loss
    return total_loss / len(data_loader)

# Define the save_checkpoint and load_checkpoint functions
def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    # Save the state of the model, optimizer, and current epoch to the specified checkpoint file
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    # Load the state of the model, optimizer, and current epoch from the specified checkpoint file
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # Return the current epoch
    return epoch

# Define the log_progress function
def log_progress(epoch, train_loss, val_loss=None):
    # Log the current training epoch and training loss
    print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.4f}', end='')
    if val_loss is not None:
        # If validation loss is provided, log it as well
        print(f' | Val Loss: {val_loss:.4f}', end='')
    print()

# Define the evaluate function
def evaluate(data_loader, model, loss_fn):
    # Set the model to evaluation mode
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            # Pass source and target data to model to obtain output logits.
            output = model(src, tgt[:-1])
            # Compute the loss between the output and target data.
            loss = loss_fn(output.view(-1,output.shape[-1]),tgt[1:].view(-1))
            # Accumulate total loss.
            total_loss += loss.item()
    # Return average validation loss.
    return total_loss / len(data_loader)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config-path', type=str, required=True, help="path to config file for train model")

    # args = parser.parse_args()

    
    # Set the file paths and other parameters
    input_file = 'English-Vietnamese translation/en_test.txt'
    output_file = 'English-Vietnamese translation/vi_test.txt'

    train_data_loader, val_data_loader, input_tokenizer, output_tokenizer = Data(input_file, 
                                                                                 output_file, 
                                                                                 batch_size)
    input_vocab_size = input_tokenizer.vocab_size()
    output_vocab_size = output_tokenizer.vocab_size()

    print(input_vocab_size, output_vocab_size)

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

    # Create a SummaryWriter instance for TensorBoard logging
    writer = SummaryWriter()

    # Train the model
    for epoch in range(num_epochs):
        train_loss = train(train_data_loader, 
                           model,
                           loss_fn,
                           optimizer(model.parameters()),
                           writer,
                           log_interval)
        val_loss = evaluate(val_data_loader,
                            model,
                            loss_fn)
        log_progress(epoch,train_loss,val_loss)
        writer.add_scalar("Loss/train_epoch",train_loss,epoch)

