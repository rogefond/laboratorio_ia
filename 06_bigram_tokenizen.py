import torch

# 1. Dataset de texto de ejemplo (nombres simples para aprender patrones)
data = """
ana
juan
pedro
maria
luis
sofia
carlos
marta
lucia
diego
""".strip().split('\n')

# Unir todo el texto para extraer el vocabulario completo
text = "\n".join(data)
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulario ({vocab_size} caracteres únicos): {chars}")

# 2. Construcción del Tokenizador
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Funciones de codificación y decodificación
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Prueba del tokenizador
ejemplo_texto = "ana"
tokens = encode(ejemplo_texto)
print(f"Texto original: '{ejemplo_texto}' -> Tokens: {tokens} -> Decodificado: '{decode(tokens)}'")

# 3. Creación de pares de entrenamiento (X -> Entrada, Y -> Objetivo)
xs, ys = [], []

for nombre in data:
    # Añadimos un carácter especial de inicio/fin usando '\n' para enseñar al modelo dónde empieza/termina una palabra
    chs = ['\n'] + list(nombre) + ['\n']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

X = torch.tensor(xs, dtype=torch.long)
Y = torch.tensor(ys, dtype=torch.long)

print(f"\nTotal de pares de entrenamiento: {len(X)}")
print(f"Primeros 5 inputs  (X): {X[:5].tolist()} -> {[itos[i.item()] for i in X[:5]]}")
print(f"Primeros 5 targets (Y): {Y[:5].tolist()} -> {[itos[i.item()] for i in Y[:5]]}")