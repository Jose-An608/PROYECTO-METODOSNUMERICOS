import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr
from .utils import limpiar_input, user_funcs, safe_funcs

# ---------------- NEWTON-RAPHSON MODIFICADO (2 VARIABLES) ----------------
def newton_modificado_2v():
    print("\n--- NEWTON-RAPHSON MODIFICADO (2 variables) ---")
    x, y = sp.symbols("x y")

    f1_str = limpiar_input(input("f1(x,y) = "))
    f2_str = limpiar_input(input("f2(x,y) = "))

    f1 = parse_expr(f1_str, {"x": x, "y": y, **user_funcs})
    f2 = parse_expr(f2_str, {"x": x, "y": y, **user_funcs})

    Fx = sp.Matrix([f1, f2])
    vars = sp.Matrix([x, y])
    J = Fx.jacobian(vars)

    f_lamb = sp.lambdify((x, y), Fx, "numpy")
    J_lamb = sp.lambdify((x, y), J, "numpy")

    x0, y0 = map(float, input("Punto inicial (x0 y0): ").split())
    tol = float(input("Tolerancia: "))
    max_iter = int(input("Máximo iteraciones: "))

    datos = []
    for i in range(max_iter):
        Fv = np.array(f_lamb(x0, y0), dtype=float).reshape(2, 1)
        Jv = np.array(J_lamb(x0, y0), dtype=float)
        try:
            delta = np.linalg.solve(Jv, -Fv)
        except np.linalg.LinAlgError:
            print("⚠ Error: matriz Jacobiana singular.")
            return
        x1, y1 = (np.array([x0, y0]) + delta.flatten())
        error = np.linalg.norm(delta)
        datos.append([i + 1, x0, y0, x1, y1, error])
        if error < tol:
            x0, y0 = x1, y1
            break
        x0, y0 = x1, y1

    tabla = pd.DataFrame(datos, columns=["Iteración", "x0", "y0", "x1", "y1", "Error"])
    print(tabla.to_string(index=False))
    print(f"\n>> Solución aproximada: x={x0}, y={y0}")

    # --- GRAFICAR EN 2D ---
    try:
        x_vals = np.linspace(x0 - 3, x0 + 3, 200)
        y_vals = np.linspace(y0 - 3, y0 + 3, 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z1 = np.vectorize(lambda x, y: eval(f1_str, {"x": x, "y": y, **safe_funcs}))(X, Y)
        Z2 = np.vectorize(lambda x, y: eval(f2_str, {"x": x, "y": y, **safe_funcs}))(X, Y)

        plt.figure(figsize=(7, 6))
        plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
        plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
        plt.scatter(x0, y0, color='black', s=70)
        plt.title("Newton-Raphson Modificado (2 variables)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)


# ---------------- NEWTON-RAPHSON MODIFICADO (3 VARIABLES) ----------------
def newton_modificado_3v():
    print("\n--- NEWTON-RAPHSON MODIFICADO (3 variables) ---")
    x, y, z = sp.symbols("x y z")

    f1_str = limpiar_input(input("f1(x,y,z) = "))
    f2_str = limpiar_input(input("f2(x,y,z) = "))
    f3_str = limpiar_input(input("f3(x,y,z) = "))

    f1 = parse_expr(f1_str, {"x": x, "y": y, "z": z, **user_funcs})
    f2 = parse_expr(f2_str, {"x": x, "y": y, "z": z, **user_funcs})
    f3 = parse_expr(f3_str, {"x": x, "y": y, "z": z, **user_funcs})

    Fx = sp.Matrix([f1, f2, f3])
    vars = sp.Matrix([x, y, z])
    J = Fx.jacobian(vars)

    f_lamb = sp.lambdify((x, y, z), Fx, "numpy")
    J_lamb = sp.lambdify((x, y, z), J, "numpy")

    x0, y0, z0 = map(float, input("Punto inicial (x0 y0 z0): ").split())
    tol = float(input("Tolerancia: "))
    max_iter = int(input("Máximo iteraciones: "))

    datos = []
    for i in range(max_iter):
        Fv = np.array(f_lamb(x0, y0, z0), dtype=float).reshape(3, 1)
        Jv = np.array(J_lamb(x0, y0, z0), dtype=float)
        try:
            delta = np.linalg.solve(Jv, -Fv)
        except np.linalg.LinAlgError:
            print("⚠ Error: matriz Jacobiana singular.")
            return
        x1, y1, z1 = (np.array([x0, y0, z0]) + delta.flatten())
        error = np.linalg.norm(delta)
        datos.append([i + 1, x0, y0, z0, x1, y1, z1, error])
        if error < tol:
            x0, y0, z0 = x1, y1, z1
            break
        x0, y0, z0 = x1, y1, z1

    tabla = pd.DataFrame(datos, columns=["Iteración", "x0", "y0", "z0", "x1", "y1", "z1", "Error"])
    print(tabla.to_string(index=False))
    print(f"\n>> Solución aproximada: x={x0}, y={y0}, z={z0}")

    # --- GRAFICAR EN 3D ---
    try:
        from mpl_toolkits.mplot3d import Axes3D
        pts = np.array([[r[1], r[2], r[3]] for r in datos])
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', color='blue', label="Trayectoria")
        ax.scatter(x0, y0, z0, color='red', s=80, label="Solución")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Newton-Raphson Modificado (3 variables)")
        ax.legend()
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)
