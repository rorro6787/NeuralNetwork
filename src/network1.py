import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import networkx as nx

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
    
    def draw_network(self):
        G = nx.Graph()
        pos = {}
        labels = {}
        node_colors = []
        node_count = 0

        # Create nodes
        for layer_index, layer_size in enumerate(self.sizes):
            for neuron_index in range(layer_size):
                node_id = node_count
                G.add_node(node_id)
                pos[node_id] = (layer_index, -neuron_index)
                
                if layer_index == 0:
                    labels[node_id] = f'input {neuron_index + 1}'
                else:
                    bias = self.biases[layer_index - 1][neuron_index, 0]
                    weight = self.weights[layer_index - 1][neuron_index]
                    weight_str = '\n|'.join([f'{w:.2f}|' for w in weight])
                    labels[node_id] = f'bias: |{bias:.2f}|\nweights:\n|{weight_str}'

                node_count += 1

                # Assign colors based on the layer
                if layer_index == 0:
                    node_colors.append('green')  # Initial layer color
                elif layer_index == len(self.sizes) - 1:
                    node_colors.append('red')  # Final layer color
                else:
                    node_colors.append('skyblue')  # Hidden layer color

        # Create edges
        node_count = 0
        for layer_index, (layer_size, next_layer_size) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            for neuron_index in range(layer_size):
                for next_neuron_index in range(next_layer_size):
                    G.add_edge(node_count + neuron_index, node_count + layer_size + next_neuron_index)
            node_count += layer_size

        # Draw the graph
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, labels=labels, with_labels=True, node_size=12000, node_color=node_colors)
        plt.show()

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
    # plot_function(relu)
    # Ejemplo de uso del forward_propagation"
    network = Network([2, 3, 1])
    #print(network.biases)
    #print(network.weights)
    #imput_data = [np.array([[1],[2]])][0]
    #print(network.forward_propagation(imput_data, sigmoid))
    print(network.weights)
    print(network.biases)
    network.draw_network()
    
if __name__ == "__main__":
    main()

