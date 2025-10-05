import os
from algebramatricial import menuma
from sistema_ecuaciones import menu_sistemas
from no_lineales import menu_no_lineales

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')  # cls 

while True:
    limpiar_pantalla()
    print("===== PROYECTO DE MÉTODOS NUMÉRICOS =====")
    print("1. Álgebra de matrices")
    print("2. Sistema de ecuaciones lineales")
    print("3. Ecuaciones no lineales")
    print("0. Salir")

    try:
        opcion = int(input("Seleccione una opción: "))
    except ValueError:
        print("Debe ingresar un número.")
        input("Presione ENTER para continuar...")
        continue

    limpiar_pantalla()  

    if opcion == 1:
        menuma()
    elif opcion == 2:
        menu_sistemas()
    elif opcion == 3:
        menu_no_lineales()
    elif opcion == 0:
        print("Saliendo...")
        break
    else:
        print(" Opción inválida.")
        input("Presione ENTER para continuar...")



