import copy
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import time
import pandas as pd
import csv


def lin_indep(a):
    q = a
    # print(a)
    c = a
    M = 0
    are_we_done = False
    x = 1
    a_square = a[0:, 0:(a.shape[0])]
    if a.shape[1] >= a.shape[0]:  # Shape[0] is the number of rows, Shape[1] is the number of columns
        b = a[0:, 0:(a.shape[0])]  # Takes the first n columns of a and checks if these are lin independent (is det 0?)
        if np.linalg.matrix_rank(b) < b.shape[0]:  # Then the columns of b are not lin independent
            for i in range(a.shape[1]):
                if are_we_done:  # If we already found a matrix with n columns that are lin independent
                    break
                for j in range(a.shape[1]):
                    if are_we_done:
                        break
                    if i != j:
                        a[:, [i, j]] = a[:, [j, i]]  # Swap column i and j
                        b = a[0:, 0:(a.shape[0])]  # Take the first n columns of a
                        if np.linalg.matrix_rank(b) < b.shape[0]:  # Is the matrix lin independent?
                            x = 1
                        else:  # The matrix is lin independent
                            # print("Lin independent matrix:")
                            M = np.linalg.det(b)  # M = determinant
                            c = b  # c is the submatrix of a with lin independent columns
                            are_we_done = True  # We have found a matrix with n columns that are lin independent
                            x = 0
        else:  # The columns of b are lin independent
            c = b
            x = 0
    else:
        c = a
        print('Error: There are more rows then columns')
    if x == 1:
        print('Error: There are no lin independent columns')
        exit(0)
    if x == 0:
        a_square = copy.deepcopy(c)
        # print(c)
        # print('DETERMINANT:')
        # print(np.linalg.det(c))  # Without rounding
        # M = abs(round(np.linalg.det(c)))  # With rounding
        # print(M)
    a_rest = a[0:, (c.shape[1]):(a.shape[1])]  # The rest of a that is not in c
    return c, a_rest


def lattice_basis_algorithm(matrix):
    c, a_rest = lin_indep(matrix)
    c = c.astype(float)
    a_rest = a_rest.astype(float)
    if np.linalg.matrix_rank(c) < c.shape[0]:
        # print(c)
        raise ValueError("Initial lattice basis is singular")
    is_x_int = True
    max_iter = 20000
    iter_count = 0
    is_it_empty = True
    tolerance = 1e-3  # For rounding
    while is_it_empty:
        is_x_int = True
        iter_count += 1
        if iter_count > max_iter:
            raise RuntimeError("Exceeded max iterations")
        if a_rest.shape[1] != 0:
            x, residuals, rank, s = np.linalg.lstsq(c, a_rest[:, 0], rcond=None)
            for i in range(len(x)):
                is_int = abs(x[i] - round(x[i]))
                toler = False
                if is_int < tolerance:
                    x[i] = round(x[i])
                    toler = True
                if not toler:
                    is_x_int = False
                    a_rest_0 = copy.deepcopy(a_rest[0:, 0])
                    a_rest = np.delete(a_rest, 0, 1)
                    a_rest = np.column_stack((a_rest, c[0:, i]))
                    d = np.zeros(c.shape[1])
                    for j in range(c.shape[1]):
                        if j != i:
                            d += c[0:, j] * math.floor(x[j])
                    c[0:, i] = a_rest_0 - (c[0:, i] * round(x[i]) + d)
                    break
            if is_x_int:
                a_rest = np.delete(a_rest, 0, 1)
        if a_rest.shape[1] == 0:
            is_it_empty = False
        if abs(round(np.linalg.det(c))) == 1:
            is_it_empty = False
    return c


def benchmark(func, matrix):
    start = time.perf_counter()
    result = func(matrix)
    end = time.perf_counter()
    return end - start, result


def run_tests(algorithm, sizes, matrix_generator, repeats=3):
    results = []
    for n in sizes:
        times = []
        failures = 0
        for _ in range(repeats):
            matrix = matrix_generator(n)
            try:
                t, _ = benchmark(algorithm, matrix)
            except Exception as e:
                print(f"Error at size {n}: {e}")
                t = float('nan')
                failures += 1
            times.append(t)
        avg_time = sum([x for x in times if not math.isnan(x)]) / (repeats - failures) if (repeats - failures) > 0 else float('nan')
        results.append((n, avg_time, failures / repeats))
    return results


def plot_results(results, label):
    sizes, times, _ = zip(*results)
    plt.plot(sizes, times, marker='o', label=label)
    plt.xlabel("Matrix size (rows = n, columns = 3n)")
    plt.ylabel("Avg runtime (seconds)")
    plt.title("Lattice Basis Algorithm Benchmark")
    plt.grid(True)
    plt.legend()


def generate_random_rectangular_matrix(n, low=-5, high=5):
    A = np.random.randint(low, high + 1, size=(n, 2*n))
    while np.linalg.matrix_rank(A[:, :n]) < n:
        A = np.random.randint(low, high + 1, size=(n, 2*n))
    return A


sizes = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
results = run_tests(lattice_basis_algorithm, sizes, generate_random_rectangular_matrix)

for n, t, fail_rate in results:
    print(f"Size {n}x{n*3}:\t{t:.6f} seconds\tFailure Rate: {fail_rate*100:.1f}%")
df = pd.DataFrame(results, columns=["Matrix Size (n)", "Avg Time (s)", "Failure Rate"])
df["Matrix Shape"] = df["Matrix Size (n)"].apply(lambda n: f"{n}x{3*n}")
df = df[["Matrix Shape", "Avg Time (s)", "Failure Rate"]]
csv_path = "C:\\Users\\lieke\\Documents\\Universiteit\\Bachelor Assignment\\Plotjes\\lattice_benchmark_results(8).csv"

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')  # <-- use semicolon
    writer.writerow(["Matrix Rows", "Matrix Columns", "Avg Runtime (s)", "Failure Rate (%)"])
    for n, t, fail_rate in results:
        writer.writerow([n, 3 * n, f"{t:.6f}", f"{fail_rate * 100:.1f}"])
plot_results(results, label="Lattice Basis Algorithm")
plt.show()




