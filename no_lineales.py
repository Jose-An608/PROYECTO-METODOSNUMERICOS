import numpy as np
from scipy.optimize import fsolve
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt

# Diccionario de funciones permitidas
user_funcs = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt
}

# =========================
# MÉTODO NEWTON-RAPHSON 2V
# =========================
def newton_raphson_2v():
    x, y = sp.symbols("x y")
    print("\nIngrese las ecuaciones en x e y (ejemplo: x**2 + y**2 - 9):")
    f1_str = input("f1(x,y) = ")
    f2_str = input("f2(x,y) = ")

    # Expresiones simbólicas
    f1_expr = parse_expr(f1_str, {"x": x, "y": y, **user_funcs})
    f2_expr = parse_expr(f2_str, {"x": x, "y": y, **user_funcs})

    # Convertimos a funciones numéricas (NumPy)
    f1 = sp.lambdify((x, y), f1_expr, "numpy")
    f2 = sp.lambdify((x, y), f2_expr, "numpy")

    def sistema(vars):
        xv, yv = vars
        return [f1(xv, yv), f2(xv, yv)]

    print("\nIngrese el punto inicial (ejemplo: 1 1):")
    guess = list(map(float, input().split()))

    # Resolvemos
    sol = fsolve(sistema, guess)
    print(">> Solución aproximada:", sol)

    # ======= Gráfica =======
    x_vals = np.linspace(sol[0] - 3, sol[0] + 3, 200)
    y_vals = np.linspace(sol[1] - 3, sol[1] + 3, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z1 = f1(X, Y)
    Z2 = f2(X, Y)

    plt.figure(figsize=(7,6))
    plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2, label='f1=0')
    plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, label='f2=0')
    plt.scatter(sol[0], sol[1], color='black', marker='o', s=80, label='Solución')
    plt.title("Intersección de f1(x,y)=0 y f2(x,y)=0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["f1(x,y)=0", "f2(x,y)=0", "Solución"])
    plt.grid(True)
    plt.show()


# =========================
# MÉTODO NEWTON-RAPHSON 3V
# =========================
def newton_raphson_3v():
    x, y, z = sp.symbols("x y z")
    print("\nIngrese las ecuaciones en x, y, z:")
    f1_str = input("f1(x,y,z) = ")
    f2_str = input("f2(x,y,z) = ")
    f3_str = input("f3(x,y,z) = ")

    # Expresiones simbólicas
    f1_expr = parse_expr(f1_str, {"x": x, "y": y, "z": z, **user_funcs})
    f2_expr = parse_expr(f2_str, {"x": x, "y": y, "z": z, **user_funcs})
    f3_expr = parse_expr(f3_str, {"x": x, "y": y, "z": z, **user_funcs})

    # Convertimos a funciones numéricas
    f1 = sp.lambdify((x, y, z), f1_expr, "numpy")
    f2 = sp.lambdify((x, y, z), f2_expr, "numpy")
    f3 = sp.lambdify((x, y, z), f3_expr, "numpy")

    def sistema(vars):
        xv, yv, zv = vars
        return [f1(xv, yv, zv), f2(xv, yv, zv), f3(xv, yv, zv)]

    print("\nIngrese el punto inicial (ejemplo: 1 1 1):")
    guess = list(map(float, input().split()))

    # Resolver con fsolve
    sol = fsolve(sistema, guess)
    print(">> Solución aproximada:", sol)

    # =============== GRÁFICA 3D ===============
    try:
        from mpl_toolkits.mplot3d import Axes3D  # necesario para 3D
        X = np.linspace(sol[0] - 2, sol[0] + 2, 40)
        Y = np.linspace(sol[1] - 2, sol[1] + 2, 40)
        X, Y = np.meshgrid(X, Y)

        # Resolver Z en función de X,Y (solo para visualizar)
        # Nota: se asume que se puede aislar z para cada ecuación
        Z1 = np.zeros_like(X)
        Z2 = np.zeros_like(X)
        Z3 = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    # Resolver z numéricamente para cada ecuación f=0
                    z1 = fsolve(lambda zz: f1(X[i,j], Y[i,j], zz), sol[2])[0]
                    z2 = fsolve(lambda zz: f2(X[i,j], Y[i,j], zz), sol[2])[0]
                    z3 = fsolve(lambda zz: f3(X[i,j], Y[i,j], zz), sol[2])[0]
                    Z1[i,j], Z2[i,j], Z3[i,j] = z1, z2, z3
                except:
                    Z1[i,j] = np.nan
                    Z2[i,j] = np.nan
                    Z3[i,j] = np.nan

        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z1, alpha=0.5, color='red', label='f1=0')
        ax.plot_surface(X, Y, Z2, alpha=0.5, color='blue', label='f2=0')
        ax.plot_surface(X, Y, Z3, alpha=0.5, color='green', label='f3=0')
        ax.scatter(sol[0], sol[1], sol[2], color='black', s=60, label='Solución')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Intersección de las superficies f1=0, f2=0, f3=0')
        plt.legend()
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar en 3D:", e)
# =========================
# MENÚ GENERAL
# =========================
def menu_no_lineales():
    while True:
        print("\n--- MÉTODOS DE ECUACIONES NO LINEALES ---")
        print("1. Newton-Raphson Multivariable (2 o 3 variables)")
        print("2. Métodos clásicos de una variable (Bisección y Secante)")
        print("3. Métodos iterativos de una variable (Punto fijo y Newton-Raphson modificado)")
        print("0. Volver al menú principal")

        opcion = int(input("Seleccione una opción: "))

        if opcion == 1:
            while True:
                print("\n--- NEWTON-RAPHSON MULTIVARIABLE ---")
                print("1. Para 2 variables")
                print("2. Para 3 variables")
                print("0. Volver")

                subop = int(input("Seleccione una opción: "))

                if subop == 1:
                    newton_raphson_2v()
                elif subop == 2:
                    newton_raphson_3v()
                elif subop == 0:
                    break
                else:
                    print("Opción inválida.")

        elif opcion == 2:
            while True:
                print("\n--- MÉTODOS CLÁSICOS DE UNA VARIABLE ---")
                print("1. Bisección")
                print("2. Secante")
                print("0. Volver")

                subop = int(input("Seleccione una opción: "))
                if subop == 1:
                    print(">> Aquí va el método de Bisección con gráfica")
                elif subop == 2:
                    print(">> Aquí va el método de Secante con gráfica")
                elif subop == 0:
                    break
                else:
                    print("Opción inválida.")

        elif opcion == 3:
            while True:
                print("\n--- MÉTODOS ITERATIVOS DE UNA VARIABLE ---")
                print("1. Punto fijo")
                print("2. Newton-Raphson modificado")
                print("0. Volver")

                subop = int(input("Seleccione una opción: "))
                if subop == 1:
                    print(">> Aquí va el método de Punto fijo")
                elif subop == 2:
                    print(">> Aquí va el Newton-Raphson modificado")
                elif subop == 0:
                    break
                else:
                    print("Opción inválida.")

        elif opcion == 0:
            break
        else:
            print("Opción inválida.")
