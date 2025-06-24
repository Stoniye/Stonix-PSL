import numpy as np
from math_functions import softmax, tanh, tanh_derivative, cross_entropy_loss, cross_entropy_loss_derivative
from public_variables import END_TOKEN, UNKNOWN_TOKEN

class RNN:  # Recurrent Neural Network
    def __init__(self, vocab_size, embedding_dim, hidden_size, learning_rate):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.W_embed = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))

        self.W_xh = np.random.uniform(-0.1, 0.1, (embedding_dim, hidden_size))
        self.W_hh = np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        self.b_h = np.zeros((1, hidden_size))

        self.W_hy = np.random.uniform(-0.1, 0.1, (hidden_size, vocab_size))
        self.b_y = np.zeros((1, vocab_size))

        self.reset_gradients()

    def reset_gradients(self):
        self.dW_embed = np.zeros_like(self.W_embed)
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_y = np.zeros_like(self.b_y)

    def forward_pass(self, input_idx, prev_hidden_state):
        embedded = self.W_embed[input_idx].reshape(1, -1)

        hidden_input = np.dot(embedded, self.W_xh) + np.dot(prev_hidden_state, self.W_hh) + self.b_h
        current_hidden_state = tanh(hidden_input)
        output_logits = np.dot(current_hidden_state, self.W_hy) + self.b_y

        cache = {
            'embedded': embedded,
            'prev_hidden_state': prev_hidden_state,
            'hidden_input': hidden_input,
            'current_hidden_state': current_hidden_state,
            'output_logits': output_logits,
            'input_idx': input_idx
        }
        return current_hidden_state, output_logits, cache

    def backward_pass(self, target_idx, cache, dnext_hidden):
        embedded = cache['embedded']
        prev_hidden_state = cache['prev_hidden_state']
        hidden_input = cache['hidden_input']
        current_hidden_state = cache['current_hidden_state']
        output_logits = cache['output_logits']
        input_idx = cache['input_idx']

        dlogits = cross_entropy_loss_derivative(softmax(output_logits).flatten(), target_idx).reshape(1, -1)

        self.dW_hy += np.dot(current_hidden_state.T, dlogits.reshape(1, -1))
        self.db_y += dlogits

        dhidden = np.dot(dlogits, self.W_hy.T)

        dhidden += dnext_hidden

        dhidden_input = dhidden * tanh_derivative(current_hidden_state)

        self.db_h += dhidden_input

        self.dW_xh += np.dot(embedded.T, dhidden_input)

        self.dW_hh += np.dot(prev_hidden_state.T, dhidden_input)

        d_hidden_prev = np.dot(dhidden_input, self.W_hh.T)

        d_embedded = np.dot(dhidden_input, self.W_xh.T)

        self.dW_embed[input_idx] += d_embedded.flatten()

        return d_hidden_prev

    def update_parameters(self):
        self.W_embed -= self.learning_rate * self.dW_embed
        self.W_xh -= self.learning_rate * self.dW_xh
        self.W_hh -= self.learning_rate * self.dW_hh
        self.b_h -= self.learning_rate * self.db_h
        self.W_hy -= self.learning_rate * self.dW_hy
        self.b_y -= self.learning_rate * self.db_y

        self.reset_gradients()

    def generate_text(self, char_to_idx, idx_to_char, start_char, num_chars_to_generate=50, temperature=1.0):
        current_hidden_state = np.zeros((1, self.hidden_size))

        generated_text = start_char
        current_input_idx = char_to_idx.get(start_char, char_to_idx[UNKNOWN_TOKEN])

        print(f"Generate text, starting with: '{start_char}'...")

        for _ in range(num_chars_to_generate):
            current_hidden_state, output_logits, _ = self.forward_pass(current_input_idx, current_hidden_state)

            probabilities = softmax(output_logits / temperature)

            next_char_idx = np.random.choice(self.vocab_size, p=probabilities.flatten())

            if next_char_idx == char_to_idx[END_TOKEN]:
                break

            generated_text += idx_to_char[next_char_idx]
            current_input_idx = next_char_idx

        return generated_text

    def train(self, epochs, x_train, y_train, char_to_index, index_to_char):
        print_every = 100

        losses = []

        for epoch in range(epochs):
            current_hidden_state = np.zeros((1, self.hidden_size))
            epoch_loss = 0

            caches = []
            target_indices_for_bptt = []

            for i in range(len(x_train)):
                input_idx = x_train[i]
                target_idx = y_train[i]

                current_hidden_state, output_logits, cache = self.forward_pass(input_idx, current_hidden_state)

                probabilities = softmax(output_logits)
                loss = cross_entropy_loss(probabilities.flatten(), target_idx)
                epoch_loss += loss

                caches.append(cache)
                target_indices_for_bptt.append(target_idx)

            dnext_hidden = np.zeros((1, self.hidden_size))

            caches.reverse()
            target_indices_for_bptt.reverse()

            for t in range(len(caches)):
                cache_t = caches[t]
                target_idx_t = target_indices_for_bptt[t]

                dnext_hidden = self.backward_pass(target_idx_t, cache_t, dnext_hidden)

            self.clip_gradients(max_norm=5.0)
            self.update_parameters()

            avg_epoch_loss = epoch_loss / len(x_train)
            losses.append(avg_epoch_loss)

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
                generated_text = self.generate_text(char_to_index, index_to_char, 'h', num_chars_to_generate=50, temperature=0.8)
                print(f"Generate Text (Epoch {epoch + 1}): {generated_text}\n")

        print(f"Training total loss: {losses[-1]:.4f}")
        print("\n--- Finished Training ---")

    def clip_gradients(self, max_norm):
        for param_grad in [self.dW_embed, self.dW_xh, self.dW_hh, self.db_h, self.dW_hy, self.db_y]:
            norm = np.linalg.norm(param_grad)
            if norm > max_norm:
                param_grad *= (max_norm / norm)

    def save_model(self, filepath):
        np.savez(filepath,
                 W_embed=self.W_embed,
                 W_xh=self.W_xh,
                 W_hh=self.W_hh,
                 b_h=self.b_h,
                 W_hy=self.W_hy,
                 b_y=self.b_y)

    def load_model(self, filepath):
        data = np.load(filepath)
        self.W_embed = data['W_embed']
        self.W_xh = data['W_xh']
        self.W_hh = data['W_hh']
        self.b_h = data['b_h']
        self.W_hy = data['W_hy']
        self.b_y = data['b_y']