from process_data import load_data, create_vocab, text_to_indices, create_training_pairs
from public_variables import DATA_PATH, MODEL_PATH, EMBEDDING_DIM, HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS
from RNN_class import RNN
from programm import chat_with_model

if __name__ == "__main__":
    data = load_data(DATA_PATH)
    char_to_index, index_to_char, vocab_size = create_vocab(data)

    text_indices = text_to_indices(data, char_to_index)
    x_train, y_train = create_training_pairs(text_indices, char_to_index)

    stonix = RNN(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, LEARNING_RATE)

    try:
        stonix.load_model(f"{MODEL_PATH}.npz")
    except FileNotFoundError:
        stonix.train(NUM_EPOCHS, x_train, y_train, char_to_index, index_to_char)

    stonix.save_model(f"{MODEL_PATH}.npz")

    chat_with_model(stonix, char_to_index, index_to_char, initial_hidden_state=None, num_chars_to_generate=50,temperature=0.8)