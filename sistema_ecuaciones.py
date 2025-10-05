import numpy as np
import os
import pandas as pd
from sistemas_ecuaciones.directos import metodo_inversa, gauss_elimination, gauss_jordan
from sistemas_ecuaciones.iterativos import jacobi, gauss_seidel

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

def leer_matriz_sistema(n):
    A = []
    print("\nIngrese los coeficientes de la matriz A:")
    for i in range(n):
        fila = []
        for j in range(n):
            while True:
                try:
                    valor = float(input(f"A[{i+1},{j+1}]: "))
                    fila.append(valor)
                    break
                except ValueError:
                    print("Debe ingresar un número válido.")
        A.append(fila)
    return np.array(A)

def leer_vector_b(n):
    b = []
    print("\nIngrese el vector b:")
    for i in range(n):
        while True:
            try:
                valor = float(input(f"b[{i+1}]: "))
                b.append(valor)
                break
            except ValueError:
                print("Debe ingresar un número válido.")
    return np.array(b)

def menu_sistemas():
    limpiar_pantalla()
    print("===== SISTEMAS DE ECUACIONES LINEALES =====")
    while True:
        try:
            n = int(input("\nIngrese el número de incógnitas: "))
            if n <= 0:
                print("Debe ingresar un número positivo.")
                continue
            break
        except ValueError:
            print("Debe ingresar un número entero.")

    A = leer_matriz_sistema(n)
    b = leer_vector_b(n)

    while True:
        limpiar_pantalla()
        print("\n===== MÉTODOS DE SOLUCIÓN =====")
        print("1. Métodos directos")
        print("2. Métodos iterativos")
        print("0. Volver")
        try:
            opcion = int(input("Seleccione una opción: "))
        except ValueError:
            print("Debe ingresar un número.")
            continue

        if opcion == 1:
            while True:
                limpiar_pantalla()
                print("\n--- Métodos Directos ---")
                print("1. Método de la inversa")
                print("2. Triangulación de Gauss")
                print("3. Gauss-Jordan")
                print("0. Volver")
                try:
                    op = int(input("Seleccione: "))
                except ValueError:
                    print("Debe ingresar un número.")
                    continue

                if op == 1:
                    metodo_inversa(A, b)
                elif op == 2:
                    resultado = gauss_elimination(A, b)
                    print("\nSolución:", resultado)
                elif op == 3:
                    resultado = gauss_jordan(A, b)
                    print("\nSolución:", resultado)
                elif op == 0:
                    break
                else:
                    print(" Opción inválida.")
                input("\nPresione ENTER para continuar...")

        elif opcion == 2:
            while True:
                limpiar_pantalla()
                print("\n--- Métodos Iterativos ---")
                print("1. Jacobi")
                print("2. Gauss-Seidel")
                print("0. Volver")
                try:
                    op = int(input("Seleccione: "))
                except ValueError:
                    print(" Debe ingresar un número.")
                    continue

                if op == 1:
                    jacobi(A, b)
                elif op == 2:
                    gauss_seidel(A, b)
                elif op == 0:
                    break
                else:
                    print(" Opción inválida.")
                input("\nPresione ENTER para continuar...")

        elif opcion == 0:
            break
        else:
            print(" Opción inválida.")
