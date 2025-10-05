import numpy as np

def metodo_inversa(A, b):
    print("\n--- Método de la Inversa ---")
    det = np.linalg.det(A)
    if det == 0:
        print(" El sistema no tiene solución única (det(A) = 0).")
        return
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    print("\nMatriz A:\n", A)
    print("\nVector b:\n", b)
    print("\nInversa de A:\n", A_inv)
    print("\nSolución del sistema (x = A⁻¹·b):\n", x)
    return x

def gauss_elimination(A, b):
    print("\n--- Método de Eliminación de Gauss ---")
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1,1)])

    for k in range(n):
        max_row = np.argmax(abs(M[k:,k])) + k
        M[[k, max_row]] = M[[max_row, k]]
        for i in range(k+1, n):
            factor = M[i][k] / M[k][k]
            M[i] = M[i] - factor * M[k]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i,i+1:n], x[i+1:n])) / M[i,i]

    return x

def gauss_jordan(A, b):
    print("\n--- Método de Gauss-Jordan ---")
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1,1)])

    for k in range(n):
        M[k] = M[k] / M[k][k]
        for i in range(n):
            if i != k:
                M[i] = M[i] - M[i][k] * M[k]

    x = M[:, -1]
    print("\nMatriz reducida a forma identidad:\n", M)
    print("\nSolución del sistema:\n", x)
    return x
