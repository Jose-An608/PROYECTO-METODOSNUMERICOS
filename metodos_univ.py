import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, lambdify, sympify

def metodo_biseccion():
    print("\n--- MÉTODO DE BISECCIÓN ---")
    x = symbols('x')
    ecuacion = input("Ingrese la función en x (ejemplo: x**3 - x - 2): ")
    f = sympify(ecuacion)
    f_lambda = lambdify(x, f, 'numpy')

    a = float(input("Ingrese el límite inferior (a): "))
    b = float(input("Ingrese el límite superior (b): "))
    tol = float(input("Ingrese la tolerancia (por ejemplo 0.0001): "))

    if f_lambda(a) * f_lambda(b) > 0:
        print("⚠ No hay cambio de signo en el intervalo.")
        return

    iteracion = 0
    c = None
    while abs(b - a) > tol:
        iteracion += 1
        c = (a + b) / 2
        if f_lambda(a) * f_lambda(c) < 0:
            b = c
        else:
            a = c
        if abs(f_lambda(c)) < tol:
            break

    print(f"\nRaíz aproximada: {c}")
    print(f"Número de iteraciones: {iteracion}")
    print(f"Tolerancia usada: {tol}")

    X = np.linspace(a - 1, b + 1, 400)
    Y = f_lambda(X)
    plt.axhline(0, color='black', lw=0.8)
    plt.plot(X, Y, label=f"f(x) = {ecuacion}")
    plt.scatter(c, f_lambda(c), color='red', label=f"Raíz ≈ {round(c,4)}")
    plt.legend()
    plt.show()


def metodo_secante():
    print("\n--- MÉTODO DE LA SECANTE ---")
    x = symbols('x')
    ecuacion = input("Ingrese la función en x (ejemplo: x**3 - x - 2): ")
    f = sympify(ecuacion)
    f_lambda = lambdify(x, f, 'numpy')

    x0 = float(input("Ingrese el primer valor inicial (x0): "))
    x1 = float(input("Ingrese el segundo valor inicial (x1): "))
    tol = float(input("Ingrese la tolerancia (por ejemplo 0.0001): "))
    max_iter = int(input("Ingrese el número máximo de iteraciones: "))

    iteracion = 0
    error = abs(x1 - x0)
    x2 = x1
    while error > tol and iteracion < max_iter:
        iteracion += 1
        f0 = f_lambda(x0)
        f1v = f_lambda(x1)

        if (f1v - f0) == 0:
            print("⚠ División por cero, el método falla.")
            return

        x2 = x1 - f1v * (x1 - x0) / (f1v - f0)
        error = abs(x2 - x1)
        x0, x1 = x1, x2

    print("\n>> Raíz aproximada:", x2)
    print(f">> Iteraciones realizadas: {iteracion}")
    print(f">> Tolerancia usada: {tol}")

    X = np.linspace(x2 - 3, x2 + 3, 400)
    Y = f_lambda(X)
    plt.axhline(0, color='black', lw=0.8)
    plt.plot(X, Y, label=f"f(x) = {ecuacion}")
    plt.scatter(x2, f_lambda(x2), color='red', s=50, label=f"Raíz ≈ {round(x2,4)}")
    plt.legend()
    plt.title("Método de la Secante")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()
