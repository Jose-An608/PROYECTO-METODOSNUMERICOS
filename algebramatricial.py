import numpy as np
import os

# ===== Funciones Auxiliares =====
def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

def leer_matriz():
    while True:
        try:
            filas = int(input("Número de filas: "))
            columnas = int(input("Número de columnas: "))
            break
        except ValueError:
            print("Ingrese un número entero válido.")
    
    matriz = []
    print("Ingrese los elementos de la matriz:")
    for i in range(filas):
        fila = []
        for j in range(columnas):
            while True:
                try:
                    valor = float(input(f"Elemento [{i+1}][{j+1}]: "))
                    break
                except ValueError:
                    print("Ingrese un número válido.")
            fila.append(valor)
        matriz.append(fila)
    return np.array(matriz)

def imprimir_matriz(matriz, titulo="Resultado"):
    print(f"\n--- {titulo} ---")
    print(matriz)

# ===== Operaciones =====
def suma():
    limpiar_pantalla()
    print("\n--- SUMA DE MATRICES ---")
    print("Primera matriz:")
    A = leer_matriz()
    print("\nSegunda matriz:")
    B = leer_matriz()
    if A.shape != B.shape:
        print(" Error: las matrices deben tener el mismo tamaño.")
    else:
        imprimir_matriz(A + B, "Suma de matrices")

def multiplicacion():
    limpiar_pantalla()
    print("\n--- MULTIPLICACIÓN DE MATRICES ---")
    print("Primera matriz:")
    A = leer_matriz()
    print("\nSegunda matriz:")
    B = leer_matriz()
    if A.shape[1] != B.shape[0]:
        print("Error: columnas de A deben coincidir con filas de B.")
    else:
        imprimir_matriz(np.dot(A, B), "Multiplicación de matrices")

def determinante():
    limpiar_pantalla()
    print("\n--- DETERMINANTE DE MATRIZ ---")
    A = leer_matriz()
    if A.shape[0] != A.shape[1]:
        print("Error: la matriz debe ser cuadrada.")
    else:
        det = np.linalg.det(A)
        print(f"\nDeterminante: {det:.6f}")

def inversa():
    limpiar_pantalla()
    print("\n--- INVERSA DE MATRIZ ---")
    A = leer_matriz()
    if A.shape[0] != A.shape[1]:
        print("Error: la matriz debe ser cuadrada.")
    else:
        try:
            inv = np.linalg.inv(A)
            imprimir_matriz(inv, "Matriz inversa")
        except np.linalg.LinAlgError:
            print("Error: la matriz no es invertible.")

# ===== Menú de Álgebra de Matrices =====
def menuma():
    while True:
        limpiar_pantalla()
        print("===== ÁLGEBRA DE MATRICES =====")
        print("1. Suma de matrices")
        print("2. Multiplicación de matrices")
        print("3. Determinante de una matriz")
        print("4. Inversa de una matriz")
        print("0. Volver")

        try:
            opcion = int(input("Seleccione una opción: "))
        except ValueError:
            print("Debe ingresar un número.")
            input("Presione ENTER para continuar...")
            continue

        if opcion == 1:
            suma()
        elif opcion == 2:
            multiplicacion()
        elif opcion == 3:
            determinante()
        elif opcion == 4:
            inversa()
        elif opcion == 0:
            break
        else:
            print("❌ Opción inválida.")
        
        input("\nPresione ENTER para continuar...")
