import numpy as np
import matplotlib.pyplot as plt
from typing import List
from typing import Any
import networkx as nx

def sigmoid(z: float) -> float:
    return 1/(1+np.exp(-z))

def relu(z: float) -> float:
    return np.maximum(0, z)

def softmax(z: float) -> float:
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def imput_vector(imput: List[float]):
    """
    Takes a network imput in the form [a_1, a_2, ..., a_n] and transforms 
    it into a vector that satisfies the dimensions: [[a_1],
                                                     [a_2]
                                                     [...],
                                                     [a_n]]
    """
    # return np.array([[value] for value in imput])
    return [[value] for value in imput]

def plot_function(function=sigmoid):
    x = np.linspace(-10, 10, 400)
    y = function(x)
    plt.plot(x, y, label=str(function.__name__) + '(x)')
    plt.title(str(function.__name__) + ' function')
    plt.xlabel('x')
    plt.ylabel(str(function.__name__) + '(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

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
        If sizes = [2, 3, 1], then weights = [List([a, b],
                                                   [c, d], 
                                                   [e, f]), List([h], [i], [j])]
        """
        self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def forward_propagation(self, imput, activation=sigmoid):
        """
        Given an imput to the neural network this function
        calculates the output given the weights, biases given the
        activation function
        """
        for b, w in zip(self.biases, self.weights):
            imput = activation(w @ imput + b)
        return imput
    
    def cuadratic_cost(self, data: List[tuple[Any, List[float]]], activation=sigmoid) -> float:
        """
        The function cuadratic_cost calculates the MSE of the network to a given imput
        sizes = [2, 4, 6], it would be a 3 layers neural network: the first layer with 2 
        neurons, the second one with 4 and the third one with 6.

        Args:
            data (List[tuple[any, List[float]]]): A list of the test objets which each one
            of them consists of an imput with type undefined and the real output that is a 
            list with as many values as output nodes has our network
        """
        cost: float = 0.0
        for kv in data:
            outp_real = kv[1] # [[1], [2]]
            outp_model = self.forward_propagation(kv[0], activation) # [[0.9], [0.98]]
            if len(outp_real) != len(outp_model):
                raise Exception("Data test dimensions not valid")
            
            model_substract = np.subtract(outp_real, outp_model) # [[1-0.9], [2-0.98]]
            model_substract_square = np.square(model_substract) # [[res1^2], [res2^2]]

            cost = np.sum(model_substract_square) # res1^2 + res2^2
        return cost/(2*len(data))      
    
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

def main():
    # plot_function()
    # Ejemplo de uso del forward_propagation"
    network = Network([3, 30, 4])
    # network.draw_network()
    # print(network.forward_propagation([[1], [2]], sigmoid))
    imput1 = imput_vector([1,2,3])
    output1 = imput_vector([1,2,3,4])
    #print(imput1-output1)
    a = [(imput1, output1)]
    # print(imput1)
    print(network.cuadratic_cost(a, activation=sigmoid))
    # print(network.biases)
    # print(network.weights)
    # imput_data = [np.array([[1],[2]])][0]
    # print(network.forward_propagation(imput_data, sigmoid))
    # print(network.weights)
    # print(network.biases)
    # network.draw_network()
    
if __name__ == "__main__":
    main()

