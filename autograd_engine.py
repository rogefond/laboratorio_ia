import math

class Value:

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Por ahora solo soportamos potencias numéricas (int/float)"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other): # other - self
        other = other if isinstance(other, Value) else Value(other)
        return other + (-self)
    """
    Guarda un valor escalar y su gradiente acumulado para diferenciación automática.
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Derivada de z = x + y -> dz/dx = 1, dz/dy = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Derivada de z = x * y -> dz/dx = y, dz/dy = x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # Derivada de d/dx(tanh(x)) = 1 - tanh(x)^2
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Ordenamiento topológico del grafo para ejecutar backprop en orden inverso
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
            
        build_topo(self)

        # El gradiente de la salida respecto a sí misma es 1.0 (dL/dL = 1)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

if __name__ == "__main__":
    # Experimento manual: Evaluación de una expresión matemática
    # f(x, y) = tanh(x * y + 2)
    x = Value(0.5)
    y = Value(2.0)
    
    # Forward Pass
    xy = x * y
    xy_plus_2 = xy + 2.0
    o = xy_plus_2.tanh()

    # Backward Pass
    o.backward()

    print("=== RESULTADO DEL AUTOGRAD ENGINE PROPIO ===")
    print(f"Salida de la función (o): {o.data:.4f}")
    print(f"Gradiente respecto a x (do/dx): {x.grad:.4f}")
    print(f"Gradiente respecto a y (do/dy): {y.grad:.4f}")