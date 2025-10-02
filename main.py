from algebramatricial import menuma
from sistema_ecuaciones import menu_sistemas

while True:

    print("PROYECTO DE METODOS NUMERICOS")
    print("1. Algebra de matrices  ")
    print("2. Sistema de ecuaciones lineales")
    print("3. Modelos con ecuaciones no lineales")
    print("4. Metodos de solucion de ecuaciones no lineales")
    print("0. Salir")

    opcion = int(input("Seleccione una opcion"))

    if opcion == 1:
        menuma()
    elif opcion == 2:
        menu_sistemas()
    elif opcion == 3:
        print("Representación de modelos con ecuaciones no lineales")
    elif opcion == 4:
        menu_metodos_nl()
    elif opcion == 0:
        break
    else:
        print("Opcion invalida.")



