import random
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Ejemplo de uso de la función sigmoide
    plot_sigmoid()
    x = 0
    result = sigmoid(x)
    print(f"El valor de la función sigmoide para x = {x} es: {result}")

if __name__ == "__main__":
    main()

def sigmoid(z):
    # Sigmpid function that we will use as the activation function
    return 1/(1+np.exp(-z))

def plot_sigmoid():
    x = np.linspace(-10, 10, 400)
    y = sigmoid(x)
    
    plt.plot(x, y, label='sigmoid(x)')
    plt.title('Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.legend()
    plt.grid(True)
    plt.show()