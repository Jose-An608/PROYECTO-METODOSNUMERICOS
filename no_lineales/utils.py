import sympy as sp
import numpy as np

def limpiar_input(expr):
    if "=" in expr:
        return expr.split("=")[1].strip()
    return expr.strip()

user_funcs = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt
}

safe_funcs = {
    "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "pi": np.pi, "e": np.e
}
