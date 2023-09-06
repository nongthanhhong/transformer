from collections import Counter
from torch.utils.data import Dataset, DataLoader
from config import *
from underthesea import text_normalize, word_tokenize
import re
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class Tokenizer:
    def __init__(self, sentences=None, save_path=None):
        if sentences is not None:
            print('=============== Init Tokenizer... ==============')
            self.vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
            for sentence in tqdm(sentences, desc="Vocabulary generating...", ncols=100):
                for word in sentence.split():
                    if word not in self.vocab:
                        self.vocab.append(word)
            self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        else:
            print('=============== Load saved Tokenizer... ==============')
            self.load(save_path)

    def tokenize(self, sentence):
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence.split()]

    def detokenize(self, tokens):
        return ' '.join([self.idx2word[token] for token in tokens])
    
    def vocab_size(self):
        return len(self.word2idx)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

# Define the TranslationDataset
class TranslationDataset(Dataset):
    def __init__(self, input_data, output_data):

        self.input_data = process_raw_sentences(raw_data = input_data, lang = 'en')
        self.output_data = process_raw_sentences(raw_data = output_data, lang = 'vi')

        # Store the source and target tokenizers
        self.input_tokenizer = Tokenizer(self.input_data)
        self.input_tokenizer.save('resources/input_tokenizer.pkl')
        self.output_tokenizer = Tokenizer(self.output_data)
        self.output_tokenizer.save('resources/output_tokenizer.pkl')

        self.sos_token = torch.tensor([self.input_tokenizer.tokenize("<sos>")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.input_tokenizer.tokenize("<eos>")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.input_tokenizer.tokenize("<pad>")], dtype=torch.int64)

    def __len__(self):

        # Return the length of the dataset (number of data points)
        return len(self.input_data)

    def __getitem__(self, idx):

        # Get the source and target data at the specified index
        input_idx, output_idx, output_target_idx = item_creator(self, 
                                                                self.input_data[idx], 
                                                                self.output_data[idx])
        
        input_mask,  output_mask= create_masks(self, 
                                               input_idx, 
                                               output_idx)

        # Double check the size of the tensors to make sure they are all seq_len long
        assert input_idx.size(0) == max_len_input
        assert output_idx.size(0) == max_len_input
        assert output_target_idx.size(0) == max_len_input


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
def create_data_loader(input_file, output_file, batch_size):
    
    # Read the source and target data from the specified files
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = f.readlines()
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = f.readlines()

    input_train, input_val, output_train, output_val = train_test_split(input_data, output_data, test_size=0.2, random_state=999)

    print('=============== Generating train data... ==============')
    # Create an instance of TranslationDataset using the provided arguments
    train_dataset = TranslationDataset(input_train, output_train)
    # Create a DataLoader with the specified batch size and shuffling enabled
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('\n\n=============== Generating validation data... ==============')
    # Create an instance of TranslationDataset using the provided arguments
    val_dataset = TranslationDataset(input_val, output_val)
    # Create a DataLoader with the specified batch size and shuffling enabled
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Return the DataLoader
    return train_data_loader, val_data_loader

def process_raw_sentences(raw_data, lang):

    # do some norm
    data = []
    if lang == "en":
        for line in tqdm(raw_data, desc="Processing raw text...", ncols=100):
            line = re.sub(r'\W+$', '', line)
            line = line.lower()
            data.append(line)
    else:
        for line in tqdm(raw_data, desc="Processing raw text...", ncols=100):
            line = re.sub(r'\W+$', '', line)
            data.append(word_tokenize(text_normalize(line), format="text"))

    return data

def item_creator(self, input_data, output_data):

    input = _padding([1] + self.input_tokenizer.tokenize(input_data) + [2])
    output = _padding([1] + self.output_tokenizer.tokenize(output_data))
    output_target = _padding(self.output_tokenizer.tokenize(output_data) + [2])

    return input, output, output_target

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

def subsequent_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def create_masks(self, input, output):

    input_mask = (input != self.pad_token).unsqueeze(0).int(), # (1, seq_len)
    output_mask = (output != self.pad_token).unsqueeze(0).int() & subsequent_mask(output.size(0)), # (1, seq_len) & (1, seq_len, seq_len),

    return input_mask[0], output_mask[0]

def Data(input_file, output_file, batch_size: int = 32):
    max_vocab_size = None # None mean take all
    

    # Create the data loader
    train_data_loader, val_data_loader = create_data_loader(input_file, 
                                    output_file,
                                    batch_size)
    

    input_tokenizer = Tokenizer(save_path="resources/input_tokenizer.pkl")
    output_tokenizer = Tokenizer(save_path="resources/output_tokenizer.pkl")

    return train_data_loader, val_data_loader, input_tokenizer, output_tokenizer
    
