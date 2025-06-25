from cpu_gpu_manager import np, USE_GPU_ENABLED, np_cpu
from math_functions import softmax
from public_variables import UNKNOWN_TOKEN, START_TOKEN, END_TOKEN

def chat_with_model(model, char_to_idx, idx_to_char, initial_hidden_state=None, num_chars_to_generate=50, temperature=0.7):

    print("\n--- Initializing Chat ---")
    print("Type 'exit' to exit")

    current_hidden_state = initial_hidden_state
    if current_hidden_state is None:
        current_hidden_state = np.zeros((1, model.hidden_size))

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chat ended")
            break

        processed_input_indices = [char_to_idx.get(char, char_to_idx[UNKNOWN_TOKEN]) for char in user_input]

        for input_idx in processed_input_indices:
            current_hidden_state, _, _ = model.forward_pass(input_idx, current_hidden_state)

        response_hidden_state = np.copy(current_hidden_state)

        next_input_for_gen = char_to_idx[START_TOKEN]

        generated_chars = []
        for _ in range(num_chars_to_generate):
            response_hidden_state, output_logits, _ = model.forward_pass(next_input_for_gen, response_hidden_state)

            probabilities = softmax(output_logits / temperature)

            if USE_GPU_ENABLED:
                probabilities_cpu = probabilities.get()
                next_char_idx = np_cpu.random.choice(model.vocab_size, size=1, p=probabilities_cpu.flatten())[0]
            else:
                next_char_idx = np.random.choice(model.vocab_size, size=1, p=probabilities.flatten())[0]

            if next_char_idx == char_to_idx[END_TOKEN]:
                break

            generated_chars.append(idx_to_char[next_char_idx])
            next_input_for_gen = next_char_idx

        model_response = "".join(generated_chars)
        print(f"Stonix: {model_response}")