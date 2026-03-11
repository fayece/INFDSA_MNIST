import numpy as np

def human_readable_dtype(dtype):
    dtype = np.dtype(dtype)
    kind = dtype.kind
    bits = dtype.itemsize * 8

    if kind == "u":
        return f"{bits}-bit unsigned integer"
    elif kind == "i":
        return f"{bits}-bit signed integer"
    elif kind == "f":
        return f"{bits}-bit float"
    elif kind == "b":
        return "Boolean"
    elif kind == "c":
        return f"{bits}-bit complex"
    elif kind == "O":
        return "Object / string"
    else:
        return str(dtype)
