import torch

print("=== VERIFICACIÓN DEL ENTORNO DE DESARROLLO ===")
print(f"Versión de PyTorch: {torch.__version__}")
print(f"Soporte CUDA (GPU NVIDIA): {torch.cuda.is_available()}")

# Asignación de tensores bidimensionales y multiplicación matricial
A = torch.randn(500, 500)
B = torch.randn(500, 500)
C = torch.matmul(A, B)

print(f"Prueba de multiplicación matricial en CPU correcta. Dimensión: {C.shape}")