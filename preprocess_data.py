from collections import Counter
from torch.utils.data import Dataset, DataLoader
from config import *
from underthesea import text_normalize, word_tokenize
import re
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import collections
import nltk
nltk.download('punkt')
import numpy as np


class Tokenizer:
    def __init__(self, sentences=None):
        if sentences is not None:
            print('=============== Init Tokenizer... ==============')
            self.vocab = collections.defaultdict(int)
            for sentence in tqdm(sentences, desc="Vocabulary generating...", ncols=100):
                for word in sentence.split('_'):
                    self.vocab[word] += 1
            self.vocab = [
                '<pad>', '<sos>', '<eos>', '<unk>',
                *[word for word, count in self.vocab.items() if count > 0]
            ]
            self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}

    def tokenize(self, sentence):
        for word in sentence.split('_'):
            yield self.word2idx.get(word, self.word2idx['<unk>'])

    def detokenize(self, tokens, mode = 1):
        """Detokenizes a list of indices or a NumPy array of indices into a string.

        Args:
            tokens: A list of indices or a NumPy array of indices.
            mode: 1 - for validation phase to calculate BLEU score, 2 - for deploy 

        Returns:
            A string containing the detokenized sentence.
        """

        sentence = []
        for token in tokens:
            if int(token) == eos_id:
                break
            sentence.append(int(token))

        if mode == 1: 
            return [self.idx2word[token] for token in sentence]
        elif mode == 2: 
            return ' '.join([self.idx2word[token] for token in sentence])
    
    def vocab_size(self):
        return len(self.vocab)
    
# Define the TranslationDataset
class TranslationDataset(Dataset):
    def __init__(self, input_data, output_data, input_tokenizer, output_tokenizer):

        self.input_data = input_data
        self.output_data = output_data
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.sos_token = torch.tensor(list(self.input_tokenizer.tokenize("<sos>")), dtype=torch.int64)
        self.eos_token = torch.tensor(list(self.input_tokenizer.tokenize("<eos>")), dtype=torch.int64)
        self.pad_token = torch.tensor(list(self.input_tokenizer.tokenize("<pad>")), dtype=torch.int64)

    def __len__(self):

        # Return the length of the dataset (number of data points)
        return len(self.input_data)

    def __getitem__(self, idx):

        # Get the source and target data at the specified index
        input_idx, output_idx, output_target_idx = item_creator(self.input_tokenizer,
                                                                self.output_tokenizer,
                                                                self.input_data[idx], 
                                                                self.output_data[idx])
        
        input_mask,  output_mask= create_masks(input_idx, 
                                               output_idx)
        


        # Double check the size of the tensors to make sure they are all seq_len long
        assert input_idx.size(0) == max_len_input
        assert output_idx.size(0) == max_len_input
        assert output_target_idx.size(0) == max_len_input

        input_idx = input_idx.to(device)
        output_idx = output_idx.to(device)
        output_target_idx = output_target_idx.to(device)
        input_mask = input_mask.to(device)
        output_mask = output_mask.to(device)

        # Return data
        return {
            "input": input_idx,  # (seq_len)
            "output": output_idx,  # (seq_len)
            "output_target": output_target_idx,  # (seq_len)
            "input_mask": input_mask, # (1, seq_len)
            "output_mask": output_mask, # (seq_len, seq_len),
            "input_text": self.input_data[idx],
            "output_text": self.output_data[idx],
        }

# Define the create_data_loader function
def create_data_loader(input_file, 
                       output_file,
                       batch_size,
                       pretrained_tokenizer = False):
    
    # Read the source and target data from the specified files
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = f.readlines()
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = f.readlines()

    input_data, output_data, input_tokenizer, output_tokenizer = Process_data(input_data, output_data, pretrained_tokenizer)

    # Create an instance of TranslationDataset using the provided arguments
    train_dataset = TranslationDataset(input_data, output_data, input_tokenizer, output_tokenizer)
    # Create a DataLoader with the specified batch size and shuffling enabled
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Return the DataLoader
    return data_loader, input_tokenizer, output_tokenizer

def Process_data(input_data, output_data, pretrained_tokenizer=False):

    input_data = process_raw_sentences(raw_data = input_data, lang = 'en')
    output_data = process_raw_sentences(raw_data = output_data, lang = 'vi')

    if pretrained_tokenizer == False: 
        # create tokenizers
        input_tokenizer = Tokenizer(sentences=input_data)
        output_tokenizer = Tokenizer(sentences=output_data)
        print("=======> Save tokenizer...")
        with open('resources/input_tokenizer.pkl', 'wb') as f:
            pickle.dump(input_tokenizer, f)
        with open('resources/output_tokenizer.pkl', 'wb') as f:
            pickle.dump(output_tokenizer, f)
    else: 
        print("=======> Load tokenizer for evaluate...")
        with open('resources/input_tokenizer.pkl', 'rb') as f:
            input_tokenizer = pickle.load(f)
        with open('resources/output_tokenizer.pkl', 'rb') as f:
            output_tokenizer = pickle.load(f)
            
    return input_data, output_data, input_tokenizer, output_tokenizer

def process_raw_sentences(raw_data, lang):

    # do some norm
    data = []
    if lang == "en":
        for line in tqdm(raw_data, desc="Processing raw text...", ncols=100):
            tokens = nltk.word_tokenize(line)
            tokens = '_'.join(token for token in tokens)
            data.append(tokens)
    else:
        for line in tqdm(raw_data, desc="Processing raw text...", ncols=100):
            tokens = word_tokenize(text_normalize(line))
            tokens = '_'.join(token for token in tokens)
            data.append(tokens)

    return data

def item_creator(input_tokenizer, output_tokenizer, input_data, output_data):

    input = _padding([sos_id] + list(input_tokenizer.tokenize(input_data)) + [eos_id])
    ouput_idx = list(output_tokenizer.tokenize(output_data))
    output = _padding([sos_id] + ouput_idx + [eos_id])
    output_target = _padding(ouput_idx + [eos_id] + [pad_id])

    return input, output, output_target

# do padding if len sentence shorter than max len, truncating if longer than max len
def _padding(tokenized_text):
    
    if len(tokenized_text) < max_len_input:
        
        tokenized_text = torch.tensor(tokenized_text, dtype=torch.int64)

        left = max_len_input - len(tokenized_text)
        padding = [pad_id] * left

        tokenized_text = torch.cat(
            [
                tokenized_text,
                torch.tensor(padding, dtype=torch.int64),
            ],
            dim=0,
        )

    else:
        tokenized_text = tokenized_text[:max_len_input]
        tokenized_text = torch.tensor(tokenized_text, dtype=torch.int64)
    
    return tokenized_text

def create_masks(input, output):

    input_mask = (input != pad_id).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
    
    subsequent_mask = torch.triu(torch.ones((1, output.size(0), output.size(0))), diagonal=1).type(torch.int) == 0
    output_mask = (output != pad_id).unsqueeze(0).int() & subsequent_mask, # (1, seq_len) & (1, seq_len, seq_len),

    return input_mask[0], output_mask[0]

def Data(train_input, train_output, val_input, val_output, batch_size: int = 32, pretrained_tokenizer = False):

    # Create the data loader
    print('\n=============== Generating Train data loader ==============')
    train_data_loader, input_tokenizer, output_tokenizer = create_data_loader(train_input,
                                                                              train_output,
                                                                              batch_size,
                                                                              pretrained_tokenizer)
    
    print('\n=============== Generating Validation data loader ==============')
    val_data_loader, _, _ = create_data_loader(val_input, 
                                               val_output,
                                               batch_size,
                                               pretrained_tokenizer)
    

    return train_data_loader, val_data_loader, input_tokenizer, output_tokenizer
    