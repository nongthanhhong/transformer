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
    print(f'\tEPOCH: {epoch} | Train Loss: {train_loss:.4f}', end='')
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
