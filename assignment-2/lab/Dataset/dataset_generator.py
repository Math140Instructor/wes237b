import os
import random
from typing import List, Tuple
import numpy as np

matrix_sizes = [
    (2, 1),
    (16, 1),
    (64, 1),
    (64, 1),
    (112, 1),
    (84, 1),
    (80, 1),
    (67, 1),
    (29, 1),
    (191, 1),
]

def write_to_file(filename : str, M : np.ndarray):
    assert(M.ndim == 1)
    with open(filename, "w") as f:
        rows = M.shape
        f.write(f"# ({rows[0]}, {1})\n")
        for row in M:
            f.write(str(row) + "\n")

for i, size in enumerate(matrix_sizes):
    A = np.random.randint(-100, 100, size[0]).T

    C = np.array([np.sum(A)])

    write_to_file(f"{i}/input0.raw", A)
    write_to_file(f"{i}/output.raw", C)

