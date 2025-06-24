# Stonix
Stonix is a hobby project of mine. I want to create my own small AI named **Stonix**.
For those who are interested, the name is simply a technical variation of my username.

# Stonix PSL
**Stonix PSL** (Pretrained Small Language Model) is the language model branch of Stonix.
I'm also working on an image recognition model for Stonix: [**Stonix Vision**](https://github.com/Stoniye/Stonix-Vision)

# Stonix PSL Documentation
### Basic Project Structure
* `datasets/` – Contains the datasets used to train Stonix
* `scripts/` – Contains all the scripts related to the project
* `trained_models/` – Contains pre-trained model files

### Configuration
You can find all the necessary configuration variables in `scripts/public_variables.py`:

* `DATA_PATH`: Path to the dataset used for training
* `MODEL_PATH`: Path where the model will be saved or loaded from (if a model with this name exists already)
* `EMBEDDING_DIM`: Size of the vector representing each character

  * Higher = better understanding of relationships, but longer training
* `HIDDEN_SIZE`: Determines context comprehension

  * Higher = better context understanding, but slower training
* `LEARNING_RATE`: Controls how much model weights are updated during training

  * Higher = faster training, but risk of overshooting
* `NUM_EPOCHS`: Number of times the model will iterate over the training data

  * Higher = better training, but longer training time

### Running the Model
To train or load your AI model, simply run: `scripts/main.py`
