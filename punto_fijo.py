import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .utils import limpiar_input, safe_funcs, user_funcs

def punto_fijo_2():
    print("\n--- Punto Fijo (2 variables) ---")
    fx = limpiar_input(input("x = "))
    fy = limpiar_input(input("y = "))

    x0,y0 = map(float,input("Punto inicial (x0 y0): ").split())
    tol = float(input("Tolerancia: "))
    max_iter = int(input("Máximo iteraciones: "))

    def g(v):
        x,y=v
        locals_dict={"x":x,"y":y,**safe_funcs}
        return np.array([eval(fx,{"__builtins__":None},locals_dict),
                         eval(fy,{"__builtins__":None},locals_dict)])

    datos=[];v=np.array([x0,y0])
    for i in range(max_iter):
        nuevo=g(v);error=np.linalg.norm(nuevo-v)
        datos.append([i+1,v[0],v[1],nuevo[0],nuevo[1],error])
        if error<tol:v=nuevo;break
        v=nuevo
    else:print("⚠ No se alcanzó la tolerancia")

    tabla=pd.DataFrame(datos,columns=["Iteración","x0","y0","x1","y1","Error"])
    print(tabla.to_string(index=False))
    print(f"\n>> Solución aproximada: x={v[0]}, y={v[1]}")

    try:
        x_vals=np.linspace(v[0]-3,v[0]+3,200)
        y_vals=np.linspace(v[1]-3,v[1]+3,200)
        X,Y=np.meshgrid(x_vals,y_vals)
        Z1=np.vectorize(lambda x,y:eval(fx,{"__builtins__":None,"x":x,"y":y,**safe_funcs}))(X,Y)-X
        Z2=np.vectorize(lambda x,y:eval(fy,{"__builtins__":None,"x":x,"y":y,**safe_funcs}))(X,Y)-Y
        plt.figure(figsize=(7,6))
        plt.contour(X,Y,Z1,levels=[0],colors='blue',linewidths=2)
        plt.contour(X,Y,Z2,levels=[0],colors='red',linewidths=2)
        plt.scatter(v[0],v[1],color='black',s=70)
        plt.title("Punto Fijo (2 variables)");plt.xlabel("x");plt.ylabel("y");plt.grid(True)
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)

def punto_fijo_3():
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from .utils import limpiar_input, safe_funcs

    print("\n--- Punto Fijo (3 variables) ---")
    fx = limpiar_input(input("x = "))
    fy = limpiar_input(input("y = "))
    fz = limpiar_input(input("z = "))

    x0, y0, z0 = map(float, input("Punto inicial (x0 y0 z0): ").split())
    tol = float(input("Tolerancia: "))
    max_iter = int(input("Máximo iteraciones: "))

    def g(v):
        x, y, z = v
        locals_dict = {"x": x, "y": y, "z": z, **safe_funcs}
        return np.array([
            eval(fx, {"__builtins__": None}, locals_dict),
            eval(fy, {"__builtins__": None}, locals_dict),
            eval(fz, {"__builtins__": None}, locals_dict)
        ])

    datos = []
    v = np.array([x0, y0, z0])
    pts = [v.copy()]  # Guardar cada iteración
    for i in range(max_iter):
        nuevo = g(v)
        error = np.linalg.norm(nuevo - v)
        datos.append([i+1, v[0], v[1], v[2], nuevo[0], nuevo[1], nuevo[2], error])
        v = nuevo
        pts.append(v.copy())
        if error < tol:
            break
    else:
        print("⚠ No se alcanzó la tolerancia")

    tabla = pd.DataFrame(datos, columns=["Iteración", "x0", "y0", "z0", "x1", "y1", "z1", "Error"])
    print(tabla.to_string(index=False))
    print(f"\n>> Solución aproximada: x={v[0]}, y={v[1]}, z={v[2]}")

    try:
        from mpl_toolkits.mplot3d import Axes3D
        pts = np.array(pts)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', color='blue', label="Trayectoria")
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color='red', s=80, label="Solución")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("Punto Fijo (3 variables) - Trayectoria")
        ax.legend()
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)
