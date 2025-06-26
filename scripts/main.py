from process_data import load_data, create_vocab, text_to_indices, create_training_pairs
from public_variables import DATA_PATH, MODEL_PATH, EMBEDDING_DIM, HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS, TEMPERATURE
from RNN_class import RNN
from programm import chat_with_model
from cpu_gpu_manager import np, USE_GPU_ENABLED

if __name__ == "__main__":

    #load and generate data for training
    data = load_data(DATA_PATH)
    char_to_index, index_to_char, vocab_size = create_vocab(data)

    text_indices_cpu = text_to_indices(data, char_to_index)
    x_train_cpu, y_train_cpu = create_training_pairs(text_indices_cpu, char_to_index)

    if USE_GPU_ENABLED:
        x_train = np.asarray(x_train_cpu)
        y_train = np.asarray(y_train_cpu)
    else:
        x_train = x_train_cpu
        y_train = y_train_cpu

    #initialize RNN model
    stonix = RNN(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, LEARNING_RATE)

    #load model parameters or train new
    try:
        stonix.load_model(f"{MODEL_PATH}.npz")
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No existing model found. Starting training...")
        stonix.train(NUM_EPOCHS, x_train, y_train, char_to_index, index_to_char)

    #start chat with model and store result if chat ends
    chatResult = chat_with_model(stonix, char_to_index, index_to_char, initial_hidden_state=None, num_chars_to_generate=50, temperature=TEMPERATURE)

    #take actions based on the chat result
    if chatResult == 'end':
        print("Chat ended")
    if chatResult == 'train':
        print("\n--- Continue Training ---")
        stonix.train(NUM_EPOCHS, x_train, y_train, char_to_index, index_to_char)