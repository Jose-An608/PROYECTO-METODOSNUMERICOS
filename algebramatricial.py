#MENU DEL ALGEBRA DE MATRICES
def menuma():

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
            
