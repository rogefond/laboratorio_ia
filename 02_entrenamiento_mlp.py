import torch
import torch.nn as nn
import torch.optim as optim

# 1. Definición de la Arquitectura
class RedRegresion(nn.Module):
    def __init__(self):
        super().__init__()
        self.capa_oculta = nn.Linear(2, 16)
        self.activacion = nn.ReLU()
        self.capa_salida = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.capa_oculta(x)
        x = self.activacion(x)
        return self.capa_salida(x)

if __name__ == "__main__":
    torch.manual_seed(42)

    # 2. Generación de Dataset Sintético (Relación lineal con ruido: Y = 3*X1 - 2*X2 + 5)
    X = torch.randn(100, 2)  # 100 muestras, 2 características
    Y_real = 3 * X[:, 0:1] - 2 * X[:, 1:2] + 5 + (torch.randn(100, 1) * 0.1)

    # 3. Instanciación del Modelo, Criterio y Optimizador
    modelo = RedRegresion()
    criterio = nn.MSELoss()
    optimizador = optim.AdamW(modelo.parameters(), lr=0.05)

    print("=== INICIANDO EL ENTRENAMIENTO EN LA CPU ===")
    
    # 4. Bucle de Entrenamiento (Training Loop)
    epochs = 100
    for epoch in range(1, epochs + 1):
        # A) Forward pass: Calcular predicción
        prediccion = modelo(X)

        # B) Calcular la pérdida (Loss)
        loss = criterio(prediccion, Y_real)

        # C) Limpiar gradientes acumulados en el paso anterior
        optimizador.zero_grad()

        # D) Backward pass: Calcular gradientes (regla de la cadena)
        loss.backward()

        # E) Actualizar pesos (W_nuevo = W_viejo - lr * gradiente)
        optimizador.step()

        # Reportar progreso cada 20 épocas
        if epoch % 20 == 0 or epoch == 1:
            print(f"Época {epoch:3d}/{epochs} | Loss (MSE): {loss.item():.6f}")

    print("\n=== EVALUACIÓN DEL APRENDIZAJE ===")
    muestra_test = torch.tensor([[1.0, 2.0]])  # Resultado teórico aproximado: 3(1) - 2(2) + 5 = 4.0
    prediccion_test = modelo(muestra_test)
    print(f"Entrada de prueba: X1=1.0, X2=2.0")
    print(f"Valor Teórico Esperado: ~4.0000")
    print(f"Predicción del Modelo:   {prediccion_test.item():.4f}")