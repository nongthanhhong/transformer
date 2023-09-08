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

    def tokenize(self, sentence):
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence.split()]

    def detokenize(self, tokens):
        return ' '.join([self.idx2word[token] for token in tokens])
    
    def vocab_size(self):
        return len(self.vocab)
    
# Define the TranslationDataset
class TranslationDataset(Dataset):
    def __init__(self, input_data, output_data, input_tokenizer, output_tokenizer):

        self.input_data = input_data
        self.output_data = output_data
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.sos_token = torch.tensor([self.input_tokenizer.tokenize("<sos>")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.input_tokenizer.tokenize("<eos>")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.input_tokenizer.tokenize("<pad>")], dtype=torch.int64)

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

def Process_data(input_data, output_data, eval):

    input_data = process_raw_sentences(raw_data = input_data, lang = 'en')
    output_data = process_raw_sentences(raw_data = output_data, lang = 'vi')

    # Store the source and target tokenizers
    input_tokenizer = Tokenizer(sentences=input_data)
    output_tokenizer = Tokenizer(sentences=output_data)
    
    if eval == False: 
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

# Define the create_data_loader function
def create_data_loader(input_file, output_file, batch_size, eval):


    print('=============== Load and process raw data ===============\n')
    # Read the source and target data from the specified files
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = f.readlines()
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = f.readlines()

    input_data, output_data, input_tokenizer, output_tokenizer = Process_data(input_data, output_data, eval)

    input_train, input_val, output_train, output_val = train_test_split(input_data, output_data, test_size=0.1, random_state=999)
    
    print('\n=============== Generating train data loader ==============')
    # Create an instance of TranslationDataset using the provided arguments
    train_dataset = TranslationDataset(input_train, output_train, input_tokenizer, output_tokenizer)
    # Create a DataLoader with the specified batch size and shuffling enabled
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('\n=============== Generating validation data loader ==============')
    # Create an instance of TranslationDataset using the provided arguments
    val_dataset = TranslationDataset(input_val, output_val, input_tokenizer, output_tokenizer)
    # Create a DataLoader with the specified batch size and shuffling enabled
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Return the DataLoader
    return train_data_loader, val_data_loader, input_tokenizer, output_tokenizer

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

def item_creator(input_tokenizer, output_tokenizer, input_data, output_data):

    input = _padding([1] + input_tokenizer.tokenize(input_data) + [2])
    output = _padding([1] + output_tokenizer.tokenize(output_data) + [2])
    output_target = _padding(output_tokenizer.tokenize(output_data) + [2])

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
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) == 0
    return mask 

def create_masks(input, output):

    input_mask = (input != pad_id).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
    
    output_mask = (output != pad_id).unsqueeze(0).int() & subsequent_mask(output.size(0)), # (1, seq_len) & (1, seq_len, seq_len),

    return input_mask[0], output_mask[0]

def Data(input_file, output_file, batch_size: int = 32, eval=False):

    # Create the data loader
    train_data_loader, val_data_loader, input_tokenizer, output_tokenizer = create_data_loader(input_file, 
                                                                                                output_file,
                                                                                                batch_size,
                                                                                                eval)

    return train_data_loader, val_data_loader, input_tokenizer, output_tokenizer
    
