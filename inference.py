from gpu_train import *
from config import *
from preprocess_data import *
import sys

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


# load model
def load_resources():

    with open('resources/input_tokenizer.pkl', 'rb') as f:
        input_tokenizer = pickle.load(f)
    with open('resources/output_tokenizer.pkl', 'rb') as f:
        output_tokenizer = pickle.load(f)

    input_vocab_size = input_tokenizer.vocab_size()
    output_vocab_size = output_tokenizer.vocab_size()

    model = Transformer(max_len=max_len_input,
                            input_vocab_size=input_vocab_size,
                            output_vocab_size=output_vocab_size,
                            num_layers=num_layers,   
                            heads=num_heads, 
                            d_model=d_model, 
                            d_ff=d_ff, 
                            dropout=drop_out_rate, 
                            bias=True).to(device)
    

    # checkpoint_path = ""
    checkpoint_path = "ckpt/"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, input_tokenizer, output_tokenizer



import re

def remove_space_after_punctuation(sentence):
    punctuation = re.compile(r' ([.!?])')
    new_sentence = punctuation.sub(r'\1', sentence)
    return new_sentence

def greedy_search(encode_input, input_mask):

    return idx_sentence


# while:
    # do predict
    # decode using greedy or beam search
# end
if __name__=="__main__":

    model, input_tokenizer, output_tokenizer = load_resources()

    while True:
        try:
            user_input = input('English: ')
        except:
            print("Cannot connect to server, please try again later!")

        sentence = "hello ! how are you ?"

        new_sentence = remove_space_after_punctuation(sentence)

        print(new_sentence)

        print(user_input)
        input_data = process_raw_sentences(raw_data = [user_input], lang = 'en')
        # output_data = process_raw_sentences(raw_data = output_data, lang = 'vi')
        
        input = _padding([1] + list(input_tokenizer.tokenize(input_data)) + [2])

        input_mask = (input != pad_id).unsqueeze(0).unsqueeze(0).int()

        print(input.shape ,input_mask.shape)
        sys.exit(0)

        output = _padding([1] + output_tokenizer.tokenize(output_data))
        output_mask

        input = input.to(device)
        output = output.to(device)
        input_mask = input_mask.to(device)
        output_mask = output_mask.to(device)

        idx_sentence = greedy_search(encode_input, input_mask)
        sentence = output_tokenizer.detokenize(idx_sentence)
        print(f'Translation: {sentence}')





