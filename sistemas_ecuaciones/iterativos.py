import numpy as np
import pandas as pd

def jacobi(A, b, tol=1e-6, max_iter=100):
    print("\n--- Método Iterativo de Jacobi ---")
    n = len(b)
    x = np.zeros(n)
    historial = []

    for it in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        historial.append([it+1] + list(x_new))

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"\n Convergió en {it+1} iteraciones")
            df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n)])
            print(df)
            return x_new
        x = x_new

    print("\n No convergió en el número máximo de iteraciones.")
    df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n)])
    print(df)
    return x

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    print("\n--- Método Iterativo de Gauss-Seidel ---")
    n = len(b)
    x = np.zeros(n)
    historial = []

    for it in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        historial.append([it+1] + list(x_new))

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"\n Convergió en {it+1} iteraciones")
            df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n)])
            print(df)
            return x_new
        x = x_new

    print("\n❌ No convergió en el número máximo de iteraciones.")
    df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n)])
    print(df)
    return x
