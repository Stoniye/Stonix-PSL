from cpu_gpu_manager import np, np_cpu, USE_GPU_ENABLED
from math_functions import softmax, tanh, tanh_derivative, cross_entropy_loss, cross_entropy_loss_derivative
from public_variables import END_TOKEN, UNKNOWN_TOKEN, MODEL_PATH


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

    def reset_gradients(self): #reset all vectors to zero vectors with right dimension
        self.dW_embed = np.zeros_like(self.W_embed)
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_y = np.zeros_like(self.b_y)

    def forward_pass(self, input_idx, prev_hidden_state):
        embedded = self.W_embed[input_idx].reshape(1, -1) #get the embedded vector for this character

        hidden_input = np.dot(embedded, self.W_xh) + np.dot(prev_hidden_state, self.W_hh) + self.b_h #calculate the new hidden vector
        current_hidden_state = tanh(hidden_input) #normalize to -1 - 1
        output_logits = np.dot(current_hidden_state, self.W_hy) + self.b_y #calcualte "raw" possabilities for the next char (this vector has the dimension (1, vocab_size))

        cache = { #store all data in one variable
            'embedded': embedded,
            'prev_hidden_state': prev_hidden_state,
            'hidden_input': hidden_input,
            'current_hidden_state': current_hidden_state,
            'output_logits': output_logits,
            'input_idx': input_idx
        }
        return current_hidden_state, output_logits, cache

    def backward_pass(self, target_idx, cache, dnext_hidden): #propagate back through the RNN to calculate the loss

        #load data from the cache variable
        embedded = cache['embedded']
        prev_hidden_state = cache['prev_hidden_state']
        hidden_input = cache['hidden_input']
        current_hidden_state = cache['current_hidden_state']
        output_logits = cache['output_logits']
        input_idx = cache['input_idx']

        dlogits = cross_entropy_loss_derivative(softmax(output_logits).flatten(), target_idx).reshape(1, -1) #compare the target_indx (char that should have been predicted) to the possibility calculated by the RNN

        #calculate gradient for the weights based on the "wrongness" of the RNN prediction
        self.dW_hy += np.dot(current_hidden_state.T, dlogits.reshape(1, -1))
        self.db_y += dlogits

        #calculate gradient for the hidden state based on the "wrongness" of the RNN prediction
        dhidden = np.dot(dlogits, self.W_hy.T)
        dhidden += dnext_hidden
        dhidden_input = dhidden * tanh_derivative(current_hidden_state)

        #calculate gradient for the bias and weights based on the "wrongness" of the RNN prediction
        self.db_h += dhidden_input
        self.dW_xh += np.dot(embedded.T, dhidden_input)
        self.dW_hh += np.dot(prev_hidden_state.T, dhidden_input)

        #calculate gradient for the next hidden stated based on the last hidden state, d_hidden_prev is the dnext_hidden in the next run
        d_hidden_prev = np.dot(dhidden_input, self.W_hh.T)
        d_embedded = np.dot(dhidden_input, self.W_xh.T)

        #tweak the gradient for the embed vector for this char based on the "wrongness"
        self.dW_embed[input_idx] += d_embedded.flatten()

        return d_hidden_prev

    def update_parameters(self): #tweak weights, bias, embed vector for loss reduction
        self.W_embed -= self.learning_rate * self.dW_embed
        self.W_xh -= self.learning_rate * self.dW_xh
        self.W_hh -= self.learning_rate * self.dW_hh
        self.b_h -= self.learning_rate * self.db_h
        self.W_hy -= self.learning_rate * self.dW_hy
        self.b_y -= self.learning_rate * self.db_y

        self.reset_gradients()

    def generate_text(self, char_to_idx, idx_to_char, start_char, num_chars_to_generate=50, temperature=1.0):
        current_hidden_state = np.zeros((1, self.hidden_size), dtype=np.float32) #create right vector dimensions

        generated_text = start_char
        current_input_idx = char_to_idx.get(start_char, char_to_idx[UNKNOWN_TOKEN]) #convert start_char to index

        print(f"Generate text, starting with: '{start_char}'...")

        for _ in range(num_chars_to_generate):
            current_hidden_state, output_logits, _ = self.forward_pass(current_input_idx, current_hidden_state) #get prediction for next char

            probabilities = softmax(output_logits / temperature) #calculate possibility for each char based on the "raw" possabilities

            #get next random next char based on the possabilities and temperature
            if USE_GPU_ENABLED:
                probabilities_cpu = probabilities.get()
                next_char_idx = np_cpu.random.choice(self.vocab_size, p=probabilities_cpu.flatten())
            else:
                next_char_idx = np.random.choice(self.vocab_size, p=probabilities.flatten())

            #end generation if end token found
            if next_char_idx == char_to_idx[END_TOKEN]:
                break

            generated_text += idx_to_char[next_char_idx] #append generated char
            current_input_idx = next_char_idx #store new char as current char

        return generated_text

    def train(self, epochs, x_train, y_train, char_to_index, index_to_char):
        print_every = 100 #print out state ever x epochs
        save_every = 1000 #save model state ever x epochs

        losses = []

        for epoch in range(epochs): #iterate through every epoch
            current_hidden_state = np.zeros((1, self.hidden_size), dtype=np.float32) #create right vector dimensions

            epoch_loss = 0

            caches = []
            target_indices_for_bptt = []

            for i in range(len(x_train)): #cycle through every char in training data
                input_idx = int(x_train[i])
                target_idx = int(y_train[i])

                #try to predict next char
                current_hidden_state, output_logits, cache = self.forward_pass(input_idx, current_hidden_state)

                probabilities = softmax(output_logits) #calculate possibility for each char based on the "raw" possabilities
                loss = cross_entropy_loss(probabilities.flatten(), target_idx) #calculate loss of the prediction ("wrongness" of the prediction)
                epoch_loss += loss

                caches.append(cache)
                target_indices_for_bptt.append(target_idx)

            dnext_hidden = np.zeros((1, self.hidden_size)) #create right vector dimensions

            #reverse Arrays
            caches.reverse()
            target_indices_for_bptt.reverse()

            for t in range(len(caches)):
                cache_t = caches[t]
                target_idx_t = target_indices_for_bptt[t]

                dnext_hidden = self.backward_pass(target_idx_t, cache_t, dnext_hidden) #propergate back to tweak weights, bias, embed

            self.clip_gradients(max_norm=5.0) #clip gradients to x value to minimize risk of overshooting
            self.update_parameters()

            #calculate epoch stats
            avg_epoch_loss = epoch_loss / len(x_train)
            losses.append(avg_epoch_loss)

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
                generated_text = self.generate_text(char_to_index, index_to_char, 'h', num_chars_to_generate=50, temperature=0.8)
                print(f"Generate Text (Epoch {epoch + 1}): {generated_text}\n")

            if (epoch + 1) % save_every == 0:
                self.save_model(MODEL_PATH)

        self.save_model(MODEL_PATH)
        print(f"Training total loss: {losses[-1]:.4f}")
        print("\n--- Finished Training ---")

    def clip_gradients(self, max_norm): #clip gradients to x value to minimize risk of overshooting
        for param_grad in [self.dW_embed, self.dW_xh, self.dW_hh, self.db_h, self.dW_hy, self.db_y]:
            norm = np.linalg.norm(param_grad)
            if norm > max_norm:
                param_grad *= (max_norm / norm)

    def transfer_parameters_to_gpu(self):
        if not USE_GPU_ENABLED: return
        print("Transferring model parameters to GPU...")
        for attr in ['W_embed', 'W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']:
            setattr(self, attr, np.asarray(getattr(self, attr)))
        for attr in ['dW_embed', 'dW_xh', 'dW_hh', 'db_h', 'dW_hy', 'db_y']:
            setattr(self, attr, np.asarray(getattr(self, attr)))

    def transfer_parameters_to_cpu(self):
        if not USE_GPU_ENABLED: return
        print("Transferring model parameters to CPU for saving...")
        for attr in ['W_embed', 'W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']:
            param = getattr(self, attr)
            if hasattr(param, 'get'):
                setattr(self, attr, param.get())
        for attr in ['dW_embed', 'dW_xh', 'dW_hh', 'db_h', 'dW_hy', 'db_y']:
            param = getattr(self, attr)
            if hasattr(param, 'get'):
                setattr(self, attr, param.get())

    def save_model(self, filepath):
        if USE_GPU_ENABLED:
            self.transfer_parameters_to_cpu()

        np_cpu.savez(filepath, W_embed=self.W_embed, W_xh=self.W_xh, W_hh=self.W_hh, b_h=self.b_h, W_hy=self.W_hy, b_y=self.b_y)

        if USE_GPU_ENABLED:
            self.transfer_parameters_to_gpu()

    def load_model(self, filepath):
        data = np_cpu.load(filepath, allow_pickle=True)

        self.W_embed = data['W_embed']
        self.W_xh = data['W_xh']
        self.W_hh = data['W_hh']
        self.b_h = data['b_h']
        self.W_hy = data['W_hy']
        self.b_y = data['b_y']

        if USE_GPU_ENABLED:
            self.transfer_parameters_to_gpu()