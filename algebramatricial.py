import numpy as np

# --- FUNCIONES DE ÁLGEBRA DE MATRICES ---

def leer_matriz():
    filas = int(input("Número de filas: "))
    columnas = int(input("Número de columnas: "))
    matriz = []
    print("Ingrese los elementos de la matriz:")
    for i in range(filas):
        fila = []
        for j in range(columnas):
            valor = float(input(f"Elemento [{i+1}][{j+1}]: "))
            fila.append(valor)
        matriz.append(fila)
    return np.array(matriz)

def suma():
    print("\n--- SUMA DE MATRICES ---")
    print("Primera matriz:")
    A = leer_matriz()
    print("Segunda matriz:")
    B = leer_matriz()
    if A.shape != B.shape:
        print("Error: las matrices deben tener el mismo tamaño")
    else:
        print("Resultado de la suma:\n", A + B)

def multiplicacion():
    print("\n--- MULTIPLICACIÓN DE MATRICES ---")
    print("Primera matriz:")
    A = leer_matriz()
    print("Segunda matriz:")
    B = leer_matriz()
    if A.shape[1] != B.shape[0]:
        print("Error: el número de columnas de A debe ser igual al número de filas de B")
    else:
        print("Resultado de la multiplicación:\n", np.dot(A, B))

def determinante():
    print("\n--- DETERMINANTE DE MATRIZ ---")
    A = leer_matriz()
    if A.shape[0] != A.shape[1]:
        print("Error: la matriz debe ser cuadrada")
    else:
        print("Determinante:", np.linalg.det(A))

def inversa():
    print("\n--- INVERSA DE MATRIZ ---")
    A = leer_matriz()
    if A.shape[0] != A.shape[1]:
        print("Error: la matriz debe ser cuadrada")
    else:
        try:
            inv = np.linalg.inv(A)
            print("Matriz inversa:\n", inv)
        except np.linalg.LinAlgError:
            print("Error: la matriz no es invertible")


# --- MENÚ PRINCIPAL DE ÁLGEBRA DE MATRICES ---
def menuma():
    while True:
        print("\nÁlgebra de matrices")
        print("1. Suma de matrices")
        print("2. Multiplicación de matrices")
        print("3. Determinante de una matriz")
        print("4. Inversa de una matriz")
        print("0. Volver")

        opcion = int(input("Seleccione una opción: "))

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
            print("Opción inválida")
