from no_lineales.newton_raphson import newton_raphson_2v, newton_raphson_3v
from no_lineales.punto_fijo import punto_fijo_2, punto_fijo_3
from no_lineales.metodos_univ import metodo_biseccion, metodo_secante

def menu_no_lineales():
    while True:
        print("\n--- MÉTODOS NO LINEALES ---")
        print("1) Newton-Raphson (2 o 3 variables)")
        print("2) Métodos clásicos de una variable")
        print("3) Punto Fijo")
        print("0) Salir")
        opcion=input("Seleccione: ").strip()

        if opcion=="1":
            sub=input("1) 2 variables\n2) 3 variables\nSeleccione: ").strip()
            if sub=="1": newton_raphson_2v()
            elif sub=="2": newton_raphson_3v()

        elif opcion=="2":
            sub=input("1) Bisección\n2) Secante\nSeleccione: ").strip()
            if sub=="1": metodo_biseccion()
            elif sub=="2": metodo_secante()

        elif opcion=="3":
            sub=input("1) 2 variables\n2) 3 variables\nSeleccione: ").strip()
            if sub=="1": punto_fijo_2()
            elif sub=="2": punto_fijo_3()

        elif opcion=="0":
            break
