import csv
import os
from math import prod


# Given a relative path to a CSV file, load it and return a list of its rows
def load_csv(filename):
    with open(os.path.normpath(filename), newline='') as csv_file:
        # Convert strings in data into numbers
        return [[float(val) if val.replace('.', '').isdigit() else val for val in row]for row in csv.reader(csv_file)]


# Saves data as a CSV.
def save_csv(filename, data):
    # Check if data is a matrix list before writing to CSV.
    if not is_matrix(data):
        return

    split_char = '\\' if os.name == 'nt' else '/'

    # Normalize path and filename
    norm_filename = os.path.normpath(filename)

    # Split file/directory name to just get directory name so it can be created if it doesn't exist.
    split_name = norm_filename.split(split_char)
    dir_name = ''

    if len(split_name) > 1:
        dir_name = split_char.join(split_name[:len(split_name) - 1])

    # Create target file directory if it doesn't exist.
    if len(dir_name) > 0 and not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # Write each row in data to the CSV.
    with open(norm_filename, 'w+') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)


# Determine if obj is a matrix. Checks if obj is a list, and if all items in it are lists.
def is_matrix(obj):
    return isinstance(obj, list) or len(list(filter(lambda x: isinstance(x, list), obj))) > 0


# Transpose columns and rows of 2D matrix
def transpose(m):
    # If m is a single row array, then nest it
    if not isinstance(m[0], list):
        m = [m]

    return [list(t) for t in zip(*m)]


# Perform some scalar function on a vector
def v_operation(operation, *v_list):
    if operation == '+':
        return [sum(x) for x in zip(*v_list)]
    if operation == '*':
        return [prod(x) for x in zip(*v_list)]
    if operation == '/':
        result_vec = []
        for x in zip(*v_list):
            result = x[0]
            for i in range(1, len(x)):
                result = result / x[i]
                result_vec.append(result)
        return result_vec


# Perform some element-wise function across multiple matrices
def m_operation(operation, *m_list):
    return [v_operation(operation, *v_list) for v_list in zip(*m_list)]


# TODO replace for s_multi?
# Perform some scalar function on a vector
def s_operation(operation, s, v, reverse_order=False):
    if operation == '+':
        return [s + x for x in v]
    if operation == '-':
        return [x - s if reverse_order else s - x for x in v]
    if operation == '*':
        return [s * x for x in v]
    if operation == '/':
        return [x / s if reverse_order else s / x for x in v]


# Perform some scalar operation between some scalar sand elements of matrix m
def sm_operation(operation, s, m):
    return [s_operation(operation, s, row) for row in m]


# Dot product of vectors v and w
def dot(v, w):
    return sum([x * y for (x, y) in zip(v, w)])


# Scalar multiply scalar c with vector v
def s_multi(c, v):
    return [c * x for x in v]


# Scalar multiply scalar c with matrix m
def sm_multi(c, m):
    return [s_multi(c, row) for row in m]


# Multiply matrix a by vector v
def mv_multi(v, a):
    return [dot(v, elm) for elm in transpose(a)]


# product of 2 compatible matrices
def m_multi(m1, m2):
    m2_zip = list(zip(*m2))
    return [[dot(z, y) for y in m2_zip] for z in m1]


def hadamard(m1, m2):
    return [[a * b for (a, b) in zip(m1_r, m2_r)] for (m1_r, m2_r) in zip(m1, m2)]
