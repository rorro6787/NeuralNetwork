import numpy as np
import matplotlib.pyplot as plt
from typing import List
import networkx as nx
import random

def sigmoid(z: float) -> float:
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z: float) -> float:
    return np.maximum(0, z)

def softmax(z: float) -> float:
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

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
    
    def forward_propagation(self, input, activation=sigmoid):
        """
        Given an imput to the neural network this function
        calculates the output given the weights, biases given the
        activation function
        """
        for b, w in zip(self.biases, self.weights):
            input = activation(w @ input + b)
        return input    
    
    def SGD(self, train_data, epochs, mini_batch_size, learning_rate):
        """
        The function SGD divides the list of train_data in a list of minibatches
        that slides the original list given a mini_batch_size. Then it calculates the 
        gradient descent of each mini_batch calling the function mini_batch_GD. It repeats
        this process epochs number of times.
        Args:
            train_data: List of tuples of the imput with the desired output
            epochs: Number of training cycles fot the network
            mini_batch_size: Size of the mini batches
            learning_rate: The learning rate of the network
        """
        n = len(train_data)
        for i in range(epochs):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # mini_batches = [ 
            #                   train_data[0:mini_batch_size]
            #                   train_data[mini_batch_size:2*mini_batch_size]
            #                   ...                 
            #                   train_data[n-mini_batch_size:n]
            # ]
            for mini_batch in mini_batches:
                self.mini_batch_GD(mini_batch, learning_rate)
    
    def mini_batch_GD(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
    
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]      

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Forward Propagation
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Output layer error
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Backpropagate the error
        for l in range(2, self.size):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            
            # Gradient of the cost function
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y) 
    
    def SGD_test(self, test_data, epochs):
        n = len(test_data)
        for i in range(epochs):
            test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
            output = sum(int(x == y) for (x, y) in test_results)
            print(f"Epoch {i}: {output} / n")
    
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
    imput1 = np.array([[1],[2],[3]])
    print(imput1)
    output1 = np.array([[1],[2],[3],[4]])
    #print(imput1-output1)
    a = [(imput1, output1)]
    print(network.weights)
    network.SGD(a, 3, 1, 1)
    print("\n\n\n")
    # print(imput1)
    print(network.weights)
    # print(network.biases)
    # print(network.weights)
    # imput_data = [np.array([[1],[2]])][0]
    # print(network.forward_propagation(imput_data, sigmoid))
    # print(network.weights)
    # print(network.biases)
    # network.draw_network()
    
if __name__ == "__main__":
    main()

