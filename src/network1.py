import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List

class Network(object):
    
    def __init__(self, sizes: List[int]):
        """
        The variable sizes contains the dimensions of our neural network. So, if
        sizes = [2, 4, 6], it would be a 3 layers neural network: the first layer with 2 
        neurons, the second one with 4 and the third one with 6.

        Args:
            sizes (List[int]): The dimensions of our neural network
        """

        # Dimensions of the neural network
        self.sizes = sizes
        # Number of layers in the neural network
        self.size = len(sizes)
        """
        List with the matrixes of biases of the neural network 
        Disclaimer: the first layer is the imput layer
        If sizes = [2, 3, 1], then biases = [List([a],
                                                  [b],
                                                  [c]), List([d])]
        """
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        """
        List with the matrixes of the weights of the neural network
        If sizes = [2, 3, 1], then weights = [List([a, b, c],
                                                   [d, e, f]), List([h],
                                                                    [i],
                                                                    [j])]
        """
        self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def forward_propagation(self, imput, activation_function):
        """
        Given an imput to the neural network this function
        calculates the output given the weights, biases given the
        activation function
        """
        for b, w in zip(self.biases, self.weights):
            imput = activation_function(w @ imput + b)
        return imput

def sigmoid(z: float) -> float:
    return 1/(1+np.exp(-z))

def relu(z: float) -> float:
    return np.maximum(0, z)

def softmax(z: float) -> float:
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def plot_function(function):
    x = np.linspace(-10, 10, 400)
    y = function(x)
    
    plt.plot(x, y, label=str(function.__name__) + '(x)')
    plt.title(str(function.__name__) + ' function')
    plt.xlabel('x')
    plt.ylabel(str(function.__name__) + '(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    plot_function(relu)
    # Ejemplo de uso del forward_propagation"
    network = Network([2, 3, 1])
    #print(network.biases)
    #print(network.weights)
    imput_data = [np.array([[1],[2]])][0]
    print(network.forward_propagation(imput_data, sigmoid))
    
if __name__ == "__main__":
    main()

