import numpy as np

def suma():
    filas = int(input("Cantidad de filas de las matrices: "))
    columnas = int(input("Cantidad de columnas de las matrices: "))

    print("Ingrese los elementos de la primera matriz:")
    matriz1 = []
    for i in range(filas):
        fila = []
        for j in range(columnas):
            valor = int(input(f"Elemento [{i+1}][{j+1}]: "))
            fila.append(valor)
        matriz1.append(fila)

    print("\nIngrese los elementos de la segunda matriz:")
    matriz2 = []
    for i in range(filas):
        fila = []
        for j in range(columnas):
            valor = int(input(f"Elemento [{i+1}][{j+1}]: "))
            fila.append(valor)
        matriz2.append(fila)

    matriz1 = np.array(matriz1)
    matriz2 = np.array(matriz2)

    suma = matriz1 + matriz2

    print("\nPrimera matriz:")
    print(matriz1)
    print("\nSegunda matriz:")
    print(matriz2)
    print("\nResultado de la suma:")
    print(suma)


def multiplicacion():
    filas1 = int(input("Cantidad de filas de la primera matriz: "))
    columnas1 = int(input("Cantidad de columnas de la primera matriz: "))

    print("Ingrese los elementos de la primera matriz:")
    matriz1 = []
    for i in range(filas1):
        fila = []
        for j in range(columnas1):
            valor = int(input(f"Elemento [{i+1}][{j+1}]: "))
            fila.append(valor)
        matriz1.append(fila)

    filas2 = int(input("Cantidad de filas de la segunda matriz: "))
    columnas2 = int(input("Cantidad de columnas de la segunda matriz: "))

    if columnas1 != filas2:
        print(f" Error: No se puede multiplicar ({filas1}x{columnas1}) con ({filas2}x{columnas2}).")
        return

    print("Ingrese los elementos de la segunda matriz:")
    matriz2 = []
    for i in range(filas2):
        fila = []
        for j in range(columnas2):
            valor = int(input(f"Elemento [{i+1}][{j+1}]: "))
            fila.append(valor)
        matriz2.append(fila)

    matriz1 = np.array(matriz1)
    matriz2 = np.array(matriz2)

    producto = np.dot(matriz1, matriz2)  

    print("\nPrimera matriz:")
    print(matriz1)
    print("\nSegunda matriz:")
    print(matriz2)
    print("\nResultado de la multiplicación:")
    print(producto)

def determinante():
    n = int(input("Ingrese el tamaño de la matriz cuadrada (n x n): "))

    matriz = []
    print("Ingrese los elementos de la matriz:")
    for i in range(n):
        fila = []
        for j in range(n):
            valor = int(input(f"Elemento [{i+1}][{j+1}]: "))
            fila.append(valor)
        matriz.append(fila)

    matriz = np.array(matriz)

    det = np.linalg.det(matriz)

    print("\nMatriz ingresada:")
    print(matriz)
    print(f"\nDeterminante: {det:.2f}")  

def inversa():
    n = int(input("Ingrese el tamaño de la matriz cuadrada (n x n): "))

    matriz = []
    print("Ingrese los elementos de la matriz:")
    for i in range(n):
        fila = []
        for j in range(n):
            valor = int(input(f"Elemento [{i+1}][{j+1}]: "))
            fila.append(valor)
        matriz.append(fila)

    matriz = np.array(matriz)
    det = np.linalg.det(matriz)

    print("\nMatriz ingresada:")
    print(matriz)

    if det == 0:
        print("\n La matriz no tiene inversa porque su determinante es 0.")
    else:
        inv = np.linalg.inv(matriz)
        print("\nInversa de la matriz:")
        print(inv)

def menu_algebra_matrices():

        print("Algebra de matrices")
        print("1. Suma de matrices")
        print("2. Multiplicacion de matrices")
        print("3. Determinante de una matriz")
        print("4. Inversa de una matriz")
        print("0. Volver")

        opcion = int(input("Seleccione una opcion "))

        if opcion == 1:
            suma()
        elif opcion == 2:
            multiplicacion()
        elif opcion == 3:
            determinante()
        elif opcion == 4:
            inversa()
        elif opcion == 0:
            return
        else:
            print("Opcion invalida")

def menu_se():
    print("Sistema de Ecuaciones Lineales (método de la inversa)")
    n = int(input("Ingrese el número de incógnitas: "))

    print("\nIngrese la matriz de coeficientes (A):")
    A = []
    for i in range(n):
        fila = []
        for j in range(n):
            valor = float(input(f"A[{i+1},{j+1}]: "))
            fila.append(valor)
        A.append(fila)

    print("\nIngrese el vector de resultados (b):")
    b = []
    for i in range(n):
        valor = float(input(f"b[{i+1}]: "))
        b.append(valor)

    A = np.array(A)
    b = np.array(b)

    det = np.linalg.det(A)
    if det == 0:
        print("\nEl sistema no tiene solución única (determinante = 0).")
    else:
        A_inv = np.linalg.inv(A) 
        x = np.dot(A_inv, b)      

        print("\nMatriz de coeficientes A:")
        print(A)
        print("\nVector b:")
        print(b)
        print("\nInversa de A:")
        print(A_inv)
        print("\nSolución del sistema (X = A^-1 * b):")
        print(x)

while True:

    print("PROYECTO DE METODOS NUMERICOS")
    print("1. Algebra de matrices  ")
    print("2. Sistema de ecuaciones lineales")
    print("3. Metodos de resolucion de sistemas")
    print("4. Modelos con ecuaciones no lineales")
    print("5. Metodos de solucion de ecuaciones no lineales")
    print("0. Salir")

    opcion = int(input("Seleccione una opcion"))

    if opcion == 1:
        menu_algebra_matrices()
    elif opcion == 2:
        menu_se()
    elif opcion == 3:
        menu_metodos_s()
    elif opcion == 4:
        print("Representación de modelos con ecuaciones no lineales")
    elif opcion == 5:
        menu_metodos_nl()
    elif opcion == 0:
        break
    else:
        print("Opcion invalida.")

