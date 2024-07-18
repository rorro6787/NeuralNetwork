import tensorflow as tf
import network1 as nt

# Cargar el dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar los datos
x_train, x_test = x_train / 255.0, x_test / 255.0
# training_data = zip(x_train, y_train)

# Ejemplo: imprimir la forma de los datos
print("Forma de x_train:", x_train.shape)
print("Forma de y_train:", y_train.shape)