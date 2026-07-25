import os
import random
from typing import List, Tuple

matrix_sizes = [
    ((2, 2), (2, 2)),
    ((16, 16), (16, 16)),
    ((64, 64), (64, 64)),
    ((64, 128), (128, 64)),
    ((112, 48), (48, 16)),
    ((84, 84), (84, 84)),
    ((80, 99), (99, 128)),
    ((67, 53), (53, 64)),
    ((29, 117), (117, 85)),
    ((191, 19), (19, 241)),
    ((2048, 2048), (2048, 2048))
]

# Function to generate input files with random int values
def generate_input_files(folder_path: str, input_idx: int, size: Tuple[int, int]):
    rows, cols = size

    input_values: List[List[int]] = []
    with open(os.path.join(folder_path, f'input{input_idx}.raw'), 'w') as file:
        file.write(f"# ({rows}, {cols})\n")
        for _ in range(rows):
            input_values.append([])
            for _ in range(cols):
                value = random.randint(-100, 100)
                input_values[-1].append(value)
                file.write(f"{value} ")
            file.write("\n")

    return input_values

# Function to generate output file with sum of int values from input0 and input1
def generate_output_file(folder_path: str, input_0: List[List[int]], input_1: List[List[int]]):
    with open(os.path.join(folder_path, 'output.raw'), 'w') as file:
        rows = len(input_0)
        cols = len(input_1[0])
        ops = len(input_0[0])

        file.write(f"# ({rows}, {cols})\n")
        output_values: List[List[int]] = []
        for row in range(rows):
            output_values.append([])
            for col in range(cols):
                sum_value = 0

                for i in range(ops):
                    sum_value += input_0[row][i] * input_1[i][col]

                output_values[-1].append(sum_value)
                file.write(f"{sum_value} ")
            file.write("\n")

# Create folders and generate files
for i, (size_0, size_1) in enumerate(matrix_sizes):
    print(f"Generating dataset {i}")
    folder_path = f"{i}"
    os.makedirs(folder_path, exist_ok=True)
    input_0 = generate_input_files(folder_path, 0, size_0)
    input_1 = generate_input_files(folder_path, 1, size_1)
    generate_output_file(folder_path, input_0, input_1)
