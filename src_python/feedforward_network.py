import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import random

print('Feedforward Network con Kernel RBF')
# Configuración de la semilla para reproducibilidad
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
print(f'Semilla: {torch.initial_seed()}')
# Parámetros
gamma = 0.1  # Parámetro del kernel RBF
C_values = [10**(-3), 10**(-2), 10**(-1), 1, 10, 100, 1000]  # Valores de regularización
L_values = [50, 100, 500, 1000, 1500, 2000]  # Neuronas en la capa oculta
print(f'Número de C: {len(C_values)}')
# Transformaciones de datos para normalización
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Cargar dataset MNIST
mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
N = len(mnist)
J = 10  # Número de clases
print(f'Número de datos: {N}')
# Partición de datos (75% entrenamiento, 25% test)
train_size = int(0.75 * N)
test_size = N - train_size
train_dataset, test_dataset = random_split(mnist, [train_size, test_size])
print(f'Número de datos de entrenamiento: {len(train_dataset)}')
# Cargar datos en DataLoader
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
print('Datos cargados')
# Obtener datos de entrenamiento y prueba
X_train, y_train = next(iter(train_loader))
X_test, y_test = next(iter(test_loader))
print('Datos obtenidos')
# Redimensionar y escalar datos
X_train = X_train.view(X_train.size(0), -1)
X_test = X_test.view(X_test.size(0), -1)
print('Datos redimensionados')
# Normalización min-max
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
print('Datos normalizados')
# One-hot encoding de etiquetas
Y_train = F.one_hot(y_train, J).float()
Y_test = F.one_hot(y_test, J).float()
print('One-hot encoding')
# Partición de validación dentro del conjunto de entrenamiento (75% entrenamiento, 25% validación)
train_val_size = int(0.75 * len(X_train))
val_size = len(X_train) - train_val_size
X_train_val, X_val = X_train[:train_val_size], X_train[train_val_size:]
Y_train_val, Y_val = Y_train[:train_val_size], Y_train[train_val_size:]
print(f'Número de datos de entrenamiento: {len(X_train_val)}')
# Matriz de rendimiento
Performance = torch.zeros(len(C_values), len(L_values))
print(f'Número de iteraciones: {len(C_values) * len(L_values)}')

# Definir kernel RBF
def rbf_kernel(X, W_hidden, gamma):
    dist = torch.cdist(X, W_hidden) ** 2
    return torch.exp(-gamma * dist)

# Búsqueda de hiperparámetros
for i, C in enumerate(C_values):
    for j, L in enumerate(L_values):
        # Generar pesos aleatorios para la capa oculta
        W_hidden = torch.randn(X_train_val.size(1), L)
        print(f'Iteración {i * len(L_values) + j + 1}: C={C}, L={L}')
        # Aplicar kernel RBF
        H = rbf_kernel(X_train_val, W_hidden.T, gamma)
        
        # Cálculo de los pesos de salida (regularización)
        W_output = torch.linalg.solve(H.T @ H + (1 / C) * torch.eye(L), H.T @ Y_train_val)
        
        # Predicciones en el conjunto de validación
        H_val = rbf_kernel(X_val, W_hidden.T, gamma)
        Y_pred_val = H_val @ W_output
        
        # CCR en validación
        Y_pred_classes = torch.argmax(Y_pred_val, dim=1)
        Y_val_classes = torch.argmax(Y_val, dim=1)
        CCR_val = (Y_pred_classes == Y_val_classes).float().mean().item()
        print(f'CCR en Validación: {CCR_val}')
        # Guardar rendimiento (CCR)
        Performance[i, j] = CCR_val

print('Búsqueda de hiperparámetros finalizada')

# Seleccionar mejores C y L
max_value = Performance.max()  # Encuentra el valor máximo de la matriz de rendimiento
max_idx = Performance.argmax()  # Encuentra el índice lineal del valor máximo
i, j = np.unravel_index(max_idx.item(), Performance.shape)

Copt = C_values[i]
Lopt = L_values[j]
print(f'Mejor C: {Copt}')
print(f'Mejor L: {Lopt}')
# Entrenamiento final con C y L óptimos
W_hidden_opt = torch.randn(X_train.size(1), Lopt)
H_opt = rbf_kernel(X_train, W_hidden_opt.T, gamma)
W_output_opt = torch.linalg.solve(H_opt.T @ H_opt + (1 / Copt) * torch.eye(Lopt), H_opt.T @ Y_train)

# Predicciones en el conjunto de prueba
H_test = rbf_kernel(X_test, W_hidden_opt.T, gamma)
Y_pred_test = H_test @ W_output_opt

# CCR en prueba
Y_pred_test_classes = torch.argmax(Y_pred_test, dim=1)
Y_test_classes = torch.argmax(Y_test, dim=1)
CCR_test = (Y_pred_test_classes == Y_test_classes).float().mean().item()

# MSE en prueba
MSE_test = F.mse_loss(Y_pred_test, Y_test).item()

# Mostrar resultados
print(f'Mejor C: {Copt}')
print(f'Mejor L: {Lopt}')
print(f'CCR en Test: {CCR_test}')
print(f'MSE en Test: {MSE_test}')
