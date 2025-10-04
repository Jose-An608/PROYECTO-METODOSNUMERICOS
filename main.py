from algebramatricial import menuma
from sistema_ecuaciones import menu_sistemas
from no_lineales import menu_no_lineales

while True:

    print("PROYECTO DE METODOS NUMERICOS")
    print("1. Algebra de matrices  ")
    print("2. Sistema de ecuaciones lineales")
    print("3. Ecuaciones no lineales ")
    print("0. Salir")

    opcion = int(input("Seleccione una opcion"))

    if opcion == 1:
        menuma()
    elif opcion == 2:
        menu_sistemas()
    elif opcion == 3:
        menu_no_lineales()
    elif opcion == 0:
        break
    else:
        print("Opcion invalida.")



