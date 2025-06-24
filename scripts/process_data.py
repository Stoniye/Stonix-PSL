import numpy as np
from public_variables import PADDING_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def create_vocab(text):
    unique_chars = sorted(list(set(text)))

    all_chars = [PADDING_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN] + unique_chars

    char_to_index = {char: i for i, char in enumerate(all_chars)}
    index_to_char = {i: char for i, char in enumerate(all_chars)}

    vocab_size = len(all_chars)

    return char_to_index, index_to_char, vocab_size

def text_to_indices(text, char_to_index):
    indices = []
    for char in text:
        indices.append(char_to_index.get(char, char_to_index[UNKNOWN_TOKEN])) # Append Unknown Token if not in vocab
    return indices

def create_training_pairs(indices, char_to_index):
    x = []
    y = []

    processed_indices = [char_to_index[START_TOKEN]] + indices

    for i in range(len(processed_indices) - 1):
        x.append(processed_indices[i])
        y.append(processed_indices[i + 1])

    return np.array(x), np.array(y)