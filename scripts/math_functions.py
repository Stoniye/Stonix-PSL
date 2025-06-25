from cpu_gpu_manager import np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x_tanh):
    return 1 - x_tanh**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x_sigmoid):
    return x_sigmoid * (1 - x_sigmoid)

def cross_entropy_loss(predictions, target_index):
    epsilon = 1e-10 #Avoid log(0)
    loss = -np.log(predictions[target_index] + epsilon)
    return loss

def cross_entropy_loss_derivative(predictions, target_index):
    gradient = np.copy(predictions)
    gradient[target_index] -= 1
    return gradient