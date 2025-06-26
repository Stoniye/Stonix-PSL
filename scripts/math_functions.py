from cpu_gpu_manager import np

def softmax(x): #calculates possibilities for every number in the vector (all sum up to 1)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def tanh(x): #tanh function
    return np.tanh(x)

def tanh_derivative(x_tanh): #derivate of the tanh function
    return 1 - x_tanh**2

def sigmoid(x): #probability function
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x_sigmoid): #derivate of the probability function
    return x_sigmoid * (1 - x_sigmoid)

def cross_entropy_loss(predictions, target_index): #calculate the loss
    epsilon = 1e-10 #Avoid log(0)
    loss = -np.log(predictions[target_index] + epsilon)
    return loss

def cross_entropy_loss_derivative(predictions, target_index): #derivate of the loss function
    gradient = np.copy(predictions)
    gradient[target_index] -= 1
    return gradient