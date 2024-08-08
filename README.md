# Basic Neural Network in Python

Welcome to the Basic Neural Network repository! This project is dedicated to building a simple neural network from scratch using basic Python. It serves as an educational tool to understand the fundamental concepts behind neural networks without relying on advanced libraries.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Introduction

This repository contains a basic implementation of a neural network, aiming to provide an understanding of how neural networks function at a fundamental level. It covers the core concepts such as forward propagation, backpropagation, and training using gradient descent.

## Features

- Basic implementation of a neural network in Python
- Forward propagation
- Backpropagation
- Training using gradient descent
- Simple and easy-to-understand code
- PDF explanation of all the code and calculus behind: [Donwload the PDF document](https://github.com/rorro6787/NeuralNetwork/blob/main/Neural_Network_Python.pdf)

## Requirements

- Python 3.x
- NumPy

## Installation

1. Clone the repository:
   
    ```sh
    git clone https://github.com/yourusername/basic-neural-network.git
    ```
3. Navigate to the project directory:
   
    ```sh
    cd basic-neural-network
    ```
5. (Optional) Create a virtual environment:
   
    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # On macOS/Linux use 'python -m venv venv
                                                   source venv/bin/activate'
    ```
7. Install the required packages:
   
    ```sh
    pip install -r requirements.txt
    ```
5. Select venv as your python interpreter (in VSC):
   
    ```sh
    > Python: Select Interpreter
    .\venv\Scripts\python.exe # On macOS/Linux use './venv/bin/python'
    ```

7. If you want to do a pull request which implies adding more dependencias, remember to update the requirements file using:
   
    ```sh
    pip freeze > requirements.txt
    ```

## Usage

1. Run the tests script to train and test the neural network:
    ```sh
    python test_minst.py
    ```

2. Modify the `network1.py` and other scripts to experiment with different network architectures, learning rates, and datasets.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## Acknowledgements

- Inspired by various tutorials and resources on neural networks and machine learning and specially by Michael Nielsen book.
