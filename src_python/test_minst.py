import numpy as np
import tensorflow as tf
import network1 as nt
import os

def load_data_wrapper(num_samples=6000):
    # Cargar el dataset MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


    indices = np.random.choice(x_train.shape[0], num_samples, replace=False)
    x_train_sample = x_train[indices]
    y_train_sample = y_train[indices]

    training_inputs = [np.reshape(x, (784, 1)) for x in x_train_sample]
    training_results = [vectorized_result(y) for y in y_train_sample]
    training_data = list(zip(training_inputs, training_results))

    test_inputs = [np.reshape(x, (784, 1)) for x in x_test]
    test_data = list(zip(test_inputs, y_test))

    # Ejemplo: imprimir la forma de los datos
    # print("Forma de x_train:", x_train.shape)
    # print("Forma de y_train:", y_train.shape)

    return (training_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    network = nt.Network([784, 30, 10])
    (training_data, test_data) = load_data_wrapper()
    #print(training_data[0][0].shape)
    network.SGD(training_data, 30, 5, 3, test_data=test_data)
    
if __name__ == "__main__":
    main()