import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from scipy.optimize import fsolve
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, sympify
import pandas as pd

# Limpia la entrada eliminando asignaciones como "x = ..." o "y = ..."
def limpiar_input(expr):
    if "=" in expr:
        return expr.split("=")[1].strip()
    return expr.strip()

# Diccionario de funciones permitidas para parse_expr (sympy)
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

    # Resolvemos con fsolve
    sol = fsolve(sistema, guess)
    print(">> Solución aproximada (fsolve):", sol)

    # ======= Gráfica =======
    try:
        x_vals = np.linspace(sol[0] - 3, sol[0] + 3, 200)
        y_vals = np.linspace(sol[1] - 3, sol[1] + 3, 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z1 = f1(X, Y)
        Z2 = f2(X, Y)

        plt.figure(figsize=(7, 6))
        plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
        plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
        plt.scatter(sol[0], sol[1], color='black', marker='o', s=80)
        plt.title("Intersección de f1(x,y)=0 y f2(x,y)=0")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["f1(x,y)=0", "f2(x,y)=0", "Solución"])
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar (newton_raphson_2v):", e)


# =========================
# MÉTODO NEWTON-RAPHSON 3V
# =========================
def newton_raphson_3v():
    x, y, z = sp.symbols("x y z")
    print("\nIngrese las ecuaciones en x, y, z:")
    f1_str = input("f1(x,y,z) = ")
    f2_str = input("f2(x,y,z) = ")
    f3_str = input("f3(x,y,z) = ")

    f1_expr = parse_expr(f1_str, {"x": x, "y": y, "z": z, **user_funcs})
    f2_expr = parse_expr(f2_str, {"x": x, "y": y, "z": z, **user_funcs})
    f3_expr = parse_expr(f3_str, {"x": x, "y": y, "z": z, **user_funcs})

    f1 = sp.lambdify((x, y, z), f1_expr, "numpy")
    f2 = sp.lambdify((x, y, z), f2_expr, "numpy")
    f3 = sp.lambdify((x, y, z), f3_expr, "numpy")

    def sistema(vars):
        xv, yv, zv = vars
        return [f1(xv, yv, zv), f2(xv, yv, zv), f3(xv, yv, zv)]

    print("\nIngrese el punto inicial (ejemplo: 1 1 1):")
    guess = list(map(float, input().split()))

    sol = fsolve(sistema, guess)
    print(">> Solución aproximada (fsolve):", sol)

    # Gráfica 3D simple (trayectoria o puntos) - no intentamos superficies costosas
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        # Hacemos una pequeña iteración de punto fijo desde el guess para tener puntos a graficar
        pts = [guess]
        v = np.array(guess, dtype=float)
        for _ in range(8):
            F = np.array(sistema(v)).flatten()
            # intento sencillo: paso corto hacia la solución fsolve
            v = v + 0.5 * (np.array(sol) - v)
            pts.append(v.copy())
        pts = np.array(pts)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o')
        ax.scatter(sol[0], sol[1], sol[2], color='red', s=60, label='fsolve solution')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title('Trayectoria / solución (Newton-Raphson 3V)')
        ax.legend()
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar en 3D (newton_raphson_3v):", e)


# =========================
# MÉTODOS DE UNA VARIABLE (Bisección, Secante, Newton modificado)
# =========================
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
        print("No hay cambio de signo en el intervalo.")
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

# ============================================
# PUNTO FIJO (2 VARIABLES) - CON TABLA Y GRÁFICA
# ============================================
def punto_fijo_2():
    print("\n--- MÉTODO DEL PUNTO FIJO (2 variables) ---")
    print("Ingrese las funciones despejadas para x e y (use sqrt, sin, cos, exp, log).")
    print("Ejemplo: x = sqrt((1+y)/2), y = sqrt((1+x)/2)")

    fx = limpiar_input(input("x = "))
    fy = limpiar_input(input("y = "))

    # entorno seguro
    safe_funcs = {
        "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "pi": np.pi, "e": np.e
    }

    x0, y0 = map(float, input("Ingrese el punto inicial (x0 y0): ").split())
    tol = float(input("Ingrese el error deseado: "))
    max_iter = int(input("Ingrese la cantidad máxima de iteraciones: "))

    def g(v):
        x, y = float(v[0]), float(v[1])
        locals_dict = {"x": x, "y": y, **safe_funcs}
        try:
            Xnew = eval(fx, {"__builtins__": None}, locals_dict)
            Ynew = eval(fy, {"__builtins__": None}, locals_dict)
        except Exception as err:
            raise ValueError(f"Error evaluando las funciones: {err}")
        return np.array([float(Xnew), float(Ynew)])

    datos = []
    v = np.array([x0, y0], dtype=float)

    for i in range(max_iter):
        try:
            nuevo = g(v)
        except ValueError as e:
            print("⚠", e)
            return

        error = np.linalg.norm(nuevo - v)
        datos.append([i + 1, v[0], v[1], nuevo[0], nuevo[1], error])

        if error < tol:
            print(f"\n✅ Convergencia alcanzada en la iteración {i + 1}.")
            v = nuevo
            break

        v = nuevo
    else:
        print("\n⚠ No se alcanzó la tolerancia dentro del número máximo de iteraciones.")

    tabla = pd.DataFrame(datos, columns=["Iteración", "x0", "y0", "x1", "y1", "Error"])
    print("\nTabla de iteraciones:\n")
    print(tabla.to_string(index=False, justify="center", col_space=10))

    print(f"\n>> Solución aproximada: x = {v[0]:.6f}, y = {v[1]:.6f}")

    # === Gráfica de las funciones g1 y g2 ===
    
    try:
        x_vals = np.linspace(v[0] - 3, v[0] + 3, 200)
        y_vals = np.linspace(v[1] - 3, v[1] + 3, 200)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Entorno seguro con funciones matemáticas permitidas
        safe_env = {
            "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "pi": np.pi, "e": np.e, "abs": abs
        }

        # Funciones vectorizadas evaluando directamente en el entorno seguro
        def eval_expr(expr, x, y):
            return eval(expr, {**safe_env, "x": x, "y": y})

        f1_eval = np.vectorize(lambda x, y: eval_expr(fx, x, y))
        f2_eval = np.vectorize(lambda x, y: eval_expr(fy, x, y))

        Z1 = f1_eval(X, Y) - X
        Z2 = f2_eval(X, Y) - Y

        plt.figure(figsize=(7, 6))
        plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
        plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
        plt.scatter(v[0], v[1], color='black', s=70, label='Solución aproximada')
        plt.title("Intersección de x = g1(x,y) y y = g2(x,y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["x = g1(x,y)", "y = g2(x,y)", "Solución"])
        plt.grid(True)
        plt.show()

    except Exception as e:
        print("⚠ No se pudo graficar las funciones:", e)

# ============================================
# PUNTO FIJO (3 VARIABLES) - CON GRÁFICA 3D SIMPLE
# ============================================
def punto_fijo_3():
    print("\n--- MÉTODO DEL PUNTO FIJO (3 variables) ---")
    print("Ingrese las funciones despejadas para x, y y z (use sqrt, sin, cos, exp, log).")

    g1_str = limpiar_input(input("x = "))
    g2_str = limpiar_input(input("y = "))
    g3_str = limpiar_input(input("z = "))


    safe_funcs = {
        "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "pi": np.pi, "e": np.e
    }

    def make_g(expr):
        def g_local(x, y, z):
            locals_dict = {"x": float(x), "y": float(y), "z": float(z), **safe_funcs}
            try:
                return float(eval(expr, {"__builtins__": None}, locals_dict))
            except Exception as err:
                raise ValueError(f"Error evaluando la función '{expr}': {err}")
        return g_local

    g1 = make_g(g1_str)
    g2 = make_g(g2_str)
    g3 = make_g(g3_str)

    x0, y0, z0 = map(float, input("Ingrese el punto inicial (x0 y0 z0): ").split())
    tol = float(input("Ingrese el error deseado: "))
    max_iter = int(input("Ingrese la cantidad máxima de iteraciones: "))

    v = np.array([x0, y0, z0], dtype=float)
    trayectoria = [v.copy()]
    iteraciones = []

    for i in range(max_iter):
        try:
            x1 = g1(v[0], v[1], v[2])
            y1 = g2(v[0], v[1], v[2])
            z1 = g3(v[0], v[1], v[2])
        except ValueError as e:
            print("⚠", e)
            return

        nuevo = np.array([x1, y1, z1], dtype=float)
        error = np.linalg.norm(nuevo - v)
        iteraciones.append([i + 1, v[0], v[1], v[2], error])
        trayectoria.append(nuevo.copy())

        if error < tol:
            print(f"\n✅ Convergencia alcanzada en la iteración {i + 1}.")
            v = nuevo
            break

        v = nuevo
    else:
        print("\n⚠ No se alcanzó la tolerancia dentro del número máximo de iteraciones.")

    print(f"\n>> Solución aproximada: x = {v[0]:.6f}, y = {v[1]:.6f}, z = {v[2]:.6f}")

    # Gráfica 3D de la trayectoria (si hay puntos)
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        traj = np.array(trayectoria)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], marker='o')
        ax.scatter(v[0], v[1], v[2], color='red', label='Punto final')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title('Trayectoria Punto Fijo (3 variables)')
        ax.legend()
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar en 3D (punto_fijo_3):", e)

def newton_raphson_modificado_2v():
    print("\n--- NEWTON-RAPHSON MODIFICADO (2 variables) ---")
    x, y = sp.symbols("x y")

    print("Ingrese las ecuaciones f1(x,y) y f2(x,y):")
    f1_str = input("f1(x,y) = ")
    f2_str = input("f2(x,y) = ")

    f1_expr = parse_expr(f1_str, {"x": x, "y": y, **user_funcs})
    f2_expr = parse_expr(f2_str, {"x": x, "y": y, **user_funcs})

    f1 = lambdify((x, y), f1_expr, "numpy")
    f2 = lambdify((x, y), f2_expr, "numpy")

    df1_dx = lambdify((x, y), sp.diff(f1_expr, x), "numpy")
    df1_dy = lambdify((x, y), sp.diff(f1_expr, y), "numpy")
    df2_dx = lambdify((x, y), sp.diff(f2_expr, x), "numpy")
    df2_dy = lambdify((x, y), sp.diff(f2_expr, y), "numpy")

    def sistema(v):
        return np.array([f1(v[0], v[1]), f2(v[0], v[1])])

    def jacobiano(v):
        return np.array([
            [df1_dx(v[0], v[1]), df1_dy(v[0], v[1])],
            [df2_dx(v[0], v[1]), df2_dy(v[0], v[1])]
        ])

    x0, y0 = map(float, input("Ingrese el punto inicial (x0 y0): ").split())
    tol = float(input("Ingrese la tolerancia: "))
    max_iter = int(input("Ingrese el número máximo de iteraciones: "))

    v = np.array([x0, y0], dtype=float)
    iteraciones = []

    for i in range(max_iter):
        F = sistema(v)
        J = jacobiano(v)

        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            print("⚠ Jacobiano singular, no se puede continuar.")
            return

        v_new = v + delta
        error = np.linalg.norm(delta)
        iteraciones.append([i + 1, v[0], v[1], v_new[0], v_new[1], error])

        if error < tol:
            v = v_new
            print(f"\n✅ Convergencia alcanzada en iteración {i+1}")
            break
        v = v_new

    tabla = pd.DataFrame(iteraciones, columns=["Iteración", "x", "y", "x_new", "y_new", "Error"])
    print("\nTabla de iteraciones:\n", tabla.to_string(index=False))
    print(f"\n>> Solución aproximada: x = {v[0]:.6f}, y = {v[1]:.6f}")

    # ======= Gráfica de las funciones =======
    try:
        x_vals = np.linspace(v[0] - 3, v[0] + 3, 200)
        y_vals = np.linspace(v[1] - 3, v[1] + 3, 200)
        X, Y = np.meshgrid(x_vals, y_vals)

        Z1 = f1(X, Y)
        Z2 = f2(X, Y)

        plt.figure(figsize=(7, 6))
        plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
        plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
        plt.scatter(v[0], v[1], color='black', marker='o', s=80, label='Solución')
        plt.title("Intersección de f1(x,y)=0 y f2(x,y)=0")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)

def newton_raphson_modificado_3v():
    print("\n--- NEWTON-RAPHSON MODIFICADO (3 variables) ---")
    x, y, z = sp.symbols("x y z")

    print("Ingrese las ecuaciones f1(x,y,z), f2(x,y,z) y f3(x,y,z):")
    f1_str = input("f1(x,y,z) = ")
    f2_str = input("f2(x,y,z) = ")
    f3_str = input("f3(x,y,z) = ")

    f1_expr = parse_expr(f1_str, {"x": x, "y": y, "z": z, **user_funcs})
    f2_expr = parse_expr(f2_str, {"x": x, "y": y, "z": z, **user_funcs})
    f3_expr = parse_expr(f3_str, {"x": x, "y": y, "z": z, **user_funcs})

    f1 = lambdify((x, y, z), f1_expr, "numpy")
    f2 = lambdify((x, y, z), f2_expr, "numpy")
    f3 = lambdify((x, y, z), f3_expr, "numpy")

    df1_dx = lambdify((x, y, z), sp.diff(f1_expr, x), "numpy")
    df1_dy = lambdify((x, y, z), sp.diff(f1_expr, y), "numpy")
    df1_dz = lambdify((x, y, z), sp.diff(f1_expr, z), "numpy")

    df2_dx = lambdify((x, y, z), sp.diff(f2_expr, x), "numpy")
    df2_dy = lambdify((x, y, z), sp.diff(f2_expr, y), "numpy")
    df2_dz = lambdify((x, y, z), sp.diff(f2_expr, z), "numpy")

    df3_dx = lambdify((x, y, z), sp.diff(f3_expr, x), "numpy")
    df3_dy = lambdify((x, y, z), sp.diff(f3_expr, y), "numpy")
    df3_dz = lambdify((x, y, z), sp.diff(f3_expr, z), "numpy")

    def sistema(v):
        return np.array([f1(v[0], v[1], v[2]),
                         f2(v[0], v[1], v[2]),
                         f3(v[0], v[1], v[2])])

    def jacobiano(v):
        return np.array([
            [df1_dx(v[0], v[1], v[2]), df1_dy(v[0], v[1], v[2]), df1_dz(v[0], v[1], v[2])],
            [df2_dx(v[0], v[1], v[2]), df2_dy(v[0], v[1], v[2]), df2_dz(v[0], v[1], v[2])],
            [df3_dx(v[0], v[1], v[2]), df3_dy(v[0], v[1], v[2]), df3_dz(v[0], v[1], v[2])]
        ])

    x0, y0, z0 = map(float, input("Ingrese el punto inicial (x0 y0 z0): ").split())
    tol = float(input("Ingrese la tolerancia: "))
    max_iter = int(input("Ingrese el número máximo de iteraciones: "))

    v = np.array([x0, y0, z0], dtype=float)
    iteraciones = []

    for i in range(max_iter):
        F = sistema(v)
        J = jacobiano(v)

        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            print("⚠ Jacobiano singular, no se puede continuar.")
            return

        v_new = v + delta
        error = np.linalg.norm(delta)
        iteraciones.append([i + 1, v[0], v[1], v[2], v_new[0], v_new[1], v_new[2], error])

        if error < tol:
            v = v_new
            print(f"\n✅ Convergencia alcanzada en iteración {i+1}")
            break
        v = v_new

    tabla = pd.DataFrame(iteraciones, columns=["Iteración", "x", "y", "z", "x_new", "y_new", "z_new", "Error"])
    print("\nTabla de iteraciones:\n", tabla.to_string(index=False))
    print(f"\n>> Solución aproximada: x = {v[0]:.6f}, y = {v[1]:.6f}, z = {v[2]:.6f}")

    # ======= Gráfica 3D de las funciones =======
    try:
        from mpl_toolkits.mplot3d import Axes3D
        x_vals = np.linspace(v[0] - 3, v[0] + 3, 30)
        y_vals = np.linspace(v[1] - 3, v[1] + 3, 30)
        X, Y = np.meshgrid(x_vals, y_vals)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        Z1 = np.array([[f1(xi, yi, v[2]) for xi in x_vals] for yi in y_vals])
        Z2 = np.array([[f2(xi, yi, v[2]) for xi in x_vals] for yi in y_vals])
        Z3 = np.array([[f3(xi, yi, v[2]) for xi in x_vals] for yi in y_vals])

        ax.plot_surface(X, Y, Z1, color='blue', alpha=0.5, label="f1=0")
        ax.plot_surface(X, Y, Z2, color='red', alpha=0.5, label="f2=0")
        ax.plot_surface(X, Y, Z3, color='green', alpha=0.5, label="f3=0")

        ax.scatter(v[0], v[1], v[2], color='black', s=50, label='Solución')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Newton-Raphson Modificado (3 variables)")
        ax.legend()
        plt.show()

    except Exception as e:
        print("⚠ No se pudo graficar:", e)


# =========================
# MENÚ GENERAL
# =========================
def menu_no_lineales():
    while True:
        print("\n--- MÉTODOS DE ECUACIONES NO LINEALES ---")
        print("1. Newton-Raphson Multivariable (2 o 3 variables)")
        print("2. Métodos clásicos de una variable (Bisección, Secante)")
        print("3. Punto Fijo Multivariable (2 o 3 variables)")
        print("0. Salir")
        opcion = input("Seleccione una opción: ").strip()

        if opcion == "1":
            print("\n1) Newton-Raphson (2 variables)\n2) Newton-Raphson (3 variables)")
            sub = input("Seleccione: ").strip()
            if sub == "1":
                newton_raphson_2v()
            elif sub == "2":
                newton_raphson_3v()

        elif opcion == "2":
            print("\n1) Bisección\n2) Secante\n")
            sub = input("Seleccione: ").strip()
            if sub == "1":
                metodo_biseccion()
            elif sub == "2":
                metodo_secante()

        elif opcion == "3":
            print("\n1) Punto Fijo (2 variables)\n2) Punto Fijo (3 variables)")
            print("3) Newton-Raphson Modificado (2 variables)\n4) Newton-Raphson Modificado (3 variables)")

            sub = input("Seleccione: ").strip()
            if sub == "1":
                punto_fijo_2()
            elif sub == "2":
                punto_fijo_3()
            elif sub == "3":
                newton_raphson_modificado_2v()
            elif sub == "4":
                newton_raphson_modificado_3v()
        elif opcion == "0":
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Intente nuevamente.")
