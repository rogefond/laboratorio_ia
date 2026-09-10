import torch
import torch.nn as nn
import torch.optim as optim

# Fijar semilla para reproducibilidad exacta
torch.manual_seed(42)

# 1. Dataset sintético convertido a Tensores de PyTorch (tipo float32)
# Entrada X: Matriz de dimensión (4, 3) -> 4 muestras, 3 características
X = torch.tensor([
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
], dtype=torch.float32)

# Etiquetas Y: Matriz de dimensión (4, 1) -> 4 objetivos
Y = torch.tensor([
    [1.0],
    [-1.0],
    [-1.0],
    [1.0]
], dtype=torch.float32)


# 2. Definición del Perceptrón Multicapa hereda de nn.Module
class MLPPyTorch(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int):
        super().__init__()
        # Capa 1: Entrada (3) -> Oculta 1 (4)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        # Capa 2: Oculta 1 (4) -> Oculta 2 (4)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Capa 3: Oculta 2 (4) -> Salida (1)
        self.fc3 = nn.Linear(hidden_dim, out_features)
        
        # Función de activación no lineal
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flujo de datos a través de las capas
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x


if __name__ == "__main__":
    # Instanciar el modelo (Entrada: 3, Ocultas: 4, Salida: 1)
    modelo = MLPPyTorch(in_features=3, hidden_dim=4, out_features=1)

    # 3. Función de pérdida y optimizador
    criterion = nn.MSELoss()  # Error Cuadrático Medio
    optimizer = optim.SGD(modelo.parameters(), lr=0.05)

    print(f"Parámetros totales a optimizar: {sum(p.numel() for p in modelo.parameters())}")
    print("\n=== ENTRENANDO RED NEURONAL EN PYTORCH ===")

    # Bucle de entrenamiento profesional (Training Loop)
    for epoch in range(100):
        # Forward pass: Cómputo de predicciones
        predictions = modelo(X)
        
        # Cálculo del Loss
        loss = criterion(predictions, Y)

        # Zero Grad: Limpieza de gradientes acumulados
        optimizer.zero_grad()

        # Backward pass: Autograd en C++
        loss.backward()

        # Step: Actualización de pesos por el optimizador
        optimizer.step()

        if epoch % 20 == 0 or epoch == 99:
            print(f"Época {epoch:2d} | Loss: {loss.item():.6f}")

    print("\n=== EVALUACIÓN FINAL DE PREDICCIONES ===")
    modelo.eval()  # Cambiar modelo a modo evaluación
    with torch.no_grad():  # Desactivar autograd para inferencia pura
        preds = modelo(X)
        for i in range(len(X)):
            x_i = X[i].tolist()
            y_real = Y[i].item()
            pred = preds[i].item()
            print(f"Entrada: {x_i} -> Real: {y_real:+.1f} | Predicción: {pred:+.4f}")