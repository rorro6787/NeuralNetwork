import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Parámetros principales
input_dim = 28 * 28  # Tamaño de entrada de imágenes MNIST
hidden_dim = 100  # Número de proyecciones aleatorias
num_classes = 10  # Número de clases en MNIST
kernel_param = 0.5  # Parámetro para el kernel gaussiano
regularization_param = 0.01  # Parámetro de regularización
batch_size = 64
num_epochs = 5

# Inicialización de pesos aleatorios para la proyección oculta (no se entrenan)
random_weights = torch.randn(input_dim, hidden_dim)

# Cargar el conjunto de datos MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Inicializar pesos de salida como parámetros de clase con referencia para optimización
output_reference_points = torch.randn(hidden_dim, num_classes, requires_grad=True)

# Definir el optimizador para actualizar solo los puntos de referencia de salida
optimizer = optim.SGD([output_reference_points], lr=0.01)

# Función de kernel gaussiano
def gaussian_kernel(x, weights, kernel_param):
    h = x @ weights  # Proyección aleatoria
    return torch.exp(-kernel_param * (h ** 2))  # Aplicación del kernel gaussiano

# Optimización de salida basada en eigenvalores
def optimize_eigen(output_reference_points, h, target_class):
    """
    Realiza la optimización de eigenvalores proyectando la clase objetivo hacia el punto de referencia
    y alejando las otras clases.
    """
    num_samples = h.shape[0]
    M_class = torch.zeros(hidden_dim, hidden_dim)
    M_non_class = torch.zeros(hidden_dim, hidden_dim)

    for i in range(num_samples):
        h_i = h[i].view(-1, 1)  # Vector columna de h para cada muestra
        if target_class[i] == 1:  # Si pertenece a la clase objetivo
            M_class += h_i @ h_i.T  # Proyecta hacia el punto de referencia
        else:
            M_non_class += h_i @ h_i.T  # Proyecta lejos del punto de referencia

    # Incorporar output_reference_points en la optimización
    M_class += torch.mm(output_reference_points.view(-1, 1), output_reference_points.view(1, -1))

    # Construcción del problema de eigenvalores regularizado
    M_class += regularization_param * torch.eye(hidden_dim)  # Regularización en M_class
    M_non_class += regularization_param * torch.eye(hidden_dim)  # Regularización en M_non_class

    # Resolución del problema de eigenvalores
    eig_vals, eig_vecs = torch.linalg.eigh(torch.inverse(M_class) @ M_non_class)
    optimal_vector = eig_vecs[:, torch.argmax(eig_vals).item()]  # Vector asociado al mayor eigenvalor
    return optimal_vector

# Entrenamiento del modelo
for epoch in range(num_epochs):
    for data, target in train_loader:
        # Preparar datos
        data = data.view(data.size(0), -1)  # Aplanar las imágenes (tamaño [batch_size, 784])
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).float()  # Codificación one-hot

        # Paso 1: Proyección aleatoria con kernel
        h = gaussian_kernel(data, random_weights, kernel_param)

        # Paso 2: Optimización para cada clase
        losses = []
        for class_index in range(num_classes):
            target_class = target_one_hot[:, class_index]
            optimal_vector = optimize_eigen(output_reference_points[:, class_index], h, target_class)
            
            # Cálculo de la pérdida: Proyectar hacia o lejos del punto de referencia
            proj_class = torch.mm(h[target_class == 1], optimal_vector.unsqueeze(1)).squeeze()
            proj_non_class = torch.mm(h[target_class == 0], optimal_vector.unsqueeze(1)).squeeze()
            
            loss_class = ((proj_class - 1) ** 2).mean()  # Muestras de la clase objetivo cerca de 1
            loss_non_class = (proj_non_class ** 2).mean()  # Muestras de otras clases cerca de 0
            losses.append(loss_class + loss_non_class)

        # Paso 3: Retropropagación y actualización
        loss = torch.stack(losses).mean()  # Promedio de pérdidas de todas las clases
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Pérdida: {loss.item():.4f}")

print("Entrenamiento completado.")
