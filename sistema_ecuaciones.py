import numpy as np
from sistemas_ecuaciones.directos import metodo_inversa, gauss_elimination, gauss_jordan
from sistemas_ecuaciones.iterativos import jacobi, gauss_seidel

def menu_sistemas():
    print("\n--- Sistemas de Ecuaciones Lineales ---")
    n = int(input("Ingrese el número de incógnitas: "))

    print("\nIngrese la matriz de coeficientes (A):")
    A = [[float(input(f"A[{i+1},{j+1}]: ")) for j in range(n)] for i in range(n)]
    b = [float(input(f"b[{i+1}]: ")) for i in range(n)]

    A = np.array(A)
    b = np.array(b)

    while True:
        print("\nElija el tipo de método:")
        print("1. Métodos directos")
        print("2. Métodos iterativos")
        print("0. Volver")  
        opcion = int(input("Opción: "))

        if opcion == 1:
            while True:
                print("\nMétodos directos:")
                print("1. Inversa")
                print("2. Triangulación de Gauss")
                print("3. Gauss-Jordan")
                print("0. Volver")
                op = int(input("Seleccione: "))

                if op == 1: metodo_inversa(A, b)
                elif op == 2: gauss_elimination(A, b)
                elif op == 3: gauss_jordan(A, b)
                elif op == 0: break
        elif opcion == 2:
            while True:
                print("\nMétodos iterativos:")
                print("1. Jacobi")
                print("2. Gauss-Seidel")
                print("0. Volver")
                op = int(input("Seleccione: "))

                if op == 1: jacobi(A, b)
                elif op == 2: gauss_seidel(A, b)
                elif op == 0: break
        elif opcion == 0:
            break
