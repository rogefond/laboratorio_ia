import torch
import torch.nn as nn

class RedDensaSimple(nn.Module):
    def __init__(self, tamano_entrada: int, tamano_oculto: int, tamano_salida: int):
        """
        Constructor: Hereda de nn.Module e inicializa las capas (parámetros).
        
        Variables matemáticas:
        - W1: Matriz de pesos (tamano_entrada x tamano_oculto)
        - b1: Vector de sesgo (tamano_oculto)
        - W2: Matriz de pesos (tamano_oculto x tamano_salida)
        - b2: Vector de sesgo (tamano_salida)
        """
        super().__init__()
        
        # Capa Lineal 1: Realiza la transformación afín Y1 = X @ W1 + b1
        self.capa_oculta = nn.Linear(tamano_entrada, tamano_oculto)
        
        # Función de activación no lineal: f(x) = max(0, x)
        self.activacion = nn.ReLU()
        
        # Capa Lineal 2: Realiza la transformación afín Y2 = Y1 @ W2 + b2
        self.capa_salida = nn.Linear(tamano_oculto, tamano_salida)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante (Forward Pass): Define el flujo de tensores.
        """
        x = self.capa_oculta(x)
        x = self.activacion(x)
        x = self.capa_salida(x)
        return x

if __name__ == "__main__":
    # Semilla para reproducibilidad de los pesos aleatorios iniciales
    torch.manual_seed(42)

    # Instanciamos la red: 4 variables de entrada, 8 neuronas ocultas, 1 variable de salida
    modelo = RedDensaSimple(tamano_entrada=4, tamano_oculto=8, tamano_salida=1)
    
    # Simulación de un lote (batch) de datos: 3 muestras, cada una con 4 características
    # Matriz X de dimensión (3, 4)
    datos_entrada = torch.randn(3, 4)
    
    # Inferencia (Forward pass invocando internamente a __call__)
    prediccion = modelo(datos_entrada)

    print("=== ARQUITECTURA DEL MODELO ===")
    print(modelo)
    
    print("\n=== PARÁMETROS ENTRENABLES INICIALIZADOS (W y b) ===")
    for nombre, parametro in modelo.named_parameters():
        print(f"Capa: {nombre} | Forma (Shape): {parametro.shape} | Requiere Gradiente: {parametro.requires_grad}")

    print("\n=== RESULTADO DE LA INFERENCIA ===")
    print("Entrada (Batch=3, Features=4):\n", datos_entrada)
    print("Salida Generada (Batch=3, Output=1):\n", prediccion)