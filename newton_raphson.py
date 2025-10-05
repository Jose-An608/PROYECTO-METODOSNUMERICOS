import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .utils import user_funcs

def newton_raphson_2v():
    print("\n--- NEWTON-RAPHSON (2 variables) ---")
    x, y = sp.symbols("x y")
    f1_str = input("f1(x,y) = ")
    f2_str = input("f2(x,y) = ")

    f1_expr = parse_expr(f1_str, {"x": x, "y": y, **user_funcs})
    f2_expr = parse_expr(f2_str, {"x": x, "y": y, **user_funcs})

    f1 = lambdify((x, y), f1_expr, "numpy")
    f2 = lambdify((x, y), f2_expr, "numpy")

    guess = list(map(float, input("Ingrese punto inicial (ej: 1 1): ").split()))

    def sistema(vars):
        return [f1(vars[0], vars[1]), f2(vars[0], vars[1])]

    from scipy.optimize import fsolve
    sol = fsolve(sistema, guess)
    print(">> Solución aproximada:", sol)

    try:
        x_vals = np.linspace(sol[0] - 3, sol[0] + 3, 200)
        y_vals = np.linspace(sol[1] - 3, sol[1] + 3, 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z1, Z2 = f1(X,Y), f2(X,Y)
        plt.figure(figsize=(7, 6))
        plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
        plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
        plt.scatter(sol[0], sol[1], color='black', s=80)
        plt.title("Newton-Raphson (2 variables)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(["f1=0","f2=0","Solución"])
        plt.grid(True); plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)

def newton_raphson_3v():
    print("\n--- NEWTON-RAPHSON (3 variables) ---")
    x, y, z = sp.symbols("x y z")
    f1_str = input("f1(x,y,z) = ")
    f2_str = input("f2(x,y,z) = ")
    f3_str = input("f3(x,y,z) = ")

    f1_expr = parse_expr(f1_str, {"x":x,"y":y,"z":z,**user_funcs})
    f2_expr = parse_expr(f2_str, {"x":x,"y":y,"z":z,**user_funcs})
    f3_expr = parse_expr(f3_str, {"x":x,"y":y,"z":z,**user_funcs})

    f1 = lambdify((x,y,z),f1_expr,"numpy")
    f2 = lambdify((x,y,z),f2_expr,"numpy")
    f3 = lambdify((x,y,z),f3_expr,"numpy")

    guess = list(map(float, input("Ingrese punto inicial (ej: 1 1 1): ").split()))

    def sistema(vars):
        return [f1(vars[0],vars[1],vars[2]),f2(vars[0],vars[1],vars[2]),f3(vars[0],vars[1],vars[2])]

    from scipy.optimize import fsolve
    sol = fsolve(sistema, guess)
    print(">> Solución aproximada:", sol)

    try:
        from mpl_toolkits.mplot3d import Axes3D
        pts = [guess]; v=np.array(guess)
        for _ in range(8):
            v = v + 0.5*(np.array(sol)-v); pts.append(v.copy())
        pts = np.array(pts)
        fig=plt.figure(figsize=(7,6))
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(pts[:,0],pts[:,1],pts[:,2],marker='o')
        ax.scatter(sol[0],sol[1],sol[2],color='red',label='Solución')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title('Newton-Raphson (3 variables)'); ax.legend()
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)
