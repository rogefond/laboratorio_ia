import random
from autograd_engine import Value


class Neuron:
    """Una neurona individual con pesos y sesgo."""
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """Una capa de neuronas conectadas en paralelo."""
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """Perceptrón Multicapa (Multi-Layer Perceptron)."""
    def __init__(self, nin: int, nouts: list[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == "__main__":
    random.seed(42)

    # Dataset sintético de prueba: 4 muestras, 3 características cada una
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    # Red: Entrada (3) -> Oculta 1 (4) -> Oculta 2 (4) -> Salida (1)
    modelo = MLP(3, [4, 4, 1])

    print(f"Parámetros totales a optimizar: {len(modelo.parameters())}")
    print("\n=== ENTRENANDO RED NEURONAL DESDE CERO ===")

    for epoch in range(100):
        # 1. Forward Pass
        ypred = [modelo(x) for x in xs]

        # 2. Loss (Error Cuadrático Medio)
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)], Value(0))

        # 3. Zero Grad
        for p in modelo.parameters():
            p.grad = 0.0

        # 4. Backward Pass
        loss.backward()

        # 5. Actualización por SGD
        learning_rate = 0.05
        for p in modelo.parameters():
            p.data -= learning_rate * p.grad

        if epoch % 20 == 0 or epoch == 99:
            print(f"Época {epoch:2d} | Loss: {loss.data:.6f}")

    print("\n=== EVALUACIÓN FINAL DE PREDICCIONES ===")
    for x, y_real in zip(xs, ys):
        pred = modelo(x)
        print(f"Entrada: {x} -> Real: {y_real:+.1f} | Predicción: {pred.data:+.4f}")