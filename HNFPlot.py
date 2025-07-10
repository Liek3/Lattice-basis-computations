import numpy as np
import time
import math
import matplotlib.pyplot as plt
import csv
import os
import copy
import openpyxl


def bounding(b_matrix, b_det):         # Given matrix and determinant, output: entries bounded by determinant
    for b_row in range(b_matrix.shape[0]):
        for b_column in range(b_matrix.shape[1]):
            T = False
            if round(b_matrix[b_row][b_column]) > b_det:
                T = True
            while T:
                b_matrix[b_row][b_column] = round(b_matrix[b_row][b_column] - b_det)
                if b_matrix[b_row][b_column] <= b_det:
                    T = False
            while b_matrix[b_row][b_column] < 0:
                b_matrix[b_row][b_column] = round(b_matrix[b_row][b_column] + b_det)
    return b_matrix


def column_operation(d):
    c_big_entry = 1e9
    c_small_entry = 1e9
    first_entry = 1e9
    second_entry = 1e9
    big_list = []
    small_list = []
    for c_entry in range(len(d[0])):
        if d[0][c_entry] > 0:
            if len(big_list) == 0:
                first_entry = c_entry
            if len(big_list) == 1:
                second_entry = c_entry
            big_list.append(d[0][c_entry])
    if big_list[0] > big_list[1]:
        c_big_entry = round(first_entry)
        c_small_entry = round(second_entry)
    if big_list[0] < big_list[1]:
        c_big_entry = round(second_entry)
        c_small_entry = round(first_entry)
    if big_list[0] == big_list[1]:
        c_big_entry = round(first_entry)
        c_small_entry = round(second_entry)
    return c_big_entry, c_small_entry


def alsocolumnopertaions(c, M):
    TorF = True
    while TorF:
        generallist = []
        for k in range(len(c[0])):
            if c[0][k] > 0:
                generallist.append(c[0][k])
        if len(generallist) > 1:            # If there is more than one nonzero entry
            i, j = column_operation(c)
            delta = math.floor(c[0][i] / c[0][j])
            c[:, i] = c[:, i] - delta * c[:, j]
            # print(c)
            bounding(c, M)
            # print(c)
        else:
            TorF = False
    return c


def hermite_algorithm(a):
    HNF = []
    q = copy.deepcopy(a)
    c = a
    M = 0
    are_we_done = False
    x = 1
    if a.shape[1] >= a.shape[0]:  # Shape[0] is the number of rows, Shape[1] is the number of columns
        b = a[0:, 0:(a.shape[0])]  # Takes the first n columns of a and checks if these are lin independent (is det 0?)
        if round(np.linalg.det(b)) == 0:  # Then the columns of b are not lin independent
            for i in range(a.shape[1]):
                if are_we_done:  # If we already found a matrix with n columns that are lin independent
                    break
                for j in range(a.shape[1]):
                    if are_we_done:
                        break
                    if i != j:
                        a[:, [i, j]] = a[:, [j, i]]  # Swap column i and j
                        b = a[0:, 0:(a.shape[0])]  # Take the first n columns of a
                        if round(np.linalg.det(b)) == 0:  # Is the matrix lin independent?
                            x = 1
                        else:  # The matrix is lin independent
                            print("Lin independent matrix:")
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
        # print(c)
        # print('DETERMINANT:')
        # print(np.linalg.det(c))  # Without rounding
        M = abs(round(np.linalg.det(c)))  # With rounding
    a = bounding(a, M)
    n = a.shape[0]
    for x in range(n):
        hello = False
        a = alsocolumnopertaions(a, M)
        for k in range(a.shape[1]):
            # print(k)
            if a[0][k] != 0:
                p = k
                hello = True
        if hello:
            HNF.append(a[:, p])
            a = np.delete(a, p, 1)
            a = np.delete(a, 0, 0)
        hello = False
    HerNorForm = np.zeros((n, n), dtype=int)
    for k in range(len(HNF)):
        HerNorForm[:, k] = np.append(np.zeros(k), HNF[k])
    HerNorForm = bounding(HerNorForm, M)
    for k in range(HerNorForm.shape[0]):
        if k > 0:
            # print(k)
            for j in range(k):
                if HerNorForm[k][k] > 0:
                    while HerNorForm[k][j] >= HerNorForm[k][k]:
                        delta = math.floor(HerNorForm[k][j] / HerNorForm[k][k])
                        HerNorForm[:, j] = HerNorForm[:, j] - delta * HerNorForm[:, k]
                    while HerNorForm[k][j] < 0:
                        delta = max(abs(math.floor(HerNorForm[k][j] / HerNorForm[k][k])), 1)
                        HerNorForm[:, j] = HerNorForm[:, j] + delta * HerNorForm[:, k]
                if HerNorForm[k][k] == 0:
                    for i in range(HerNorForm.shape[1]):
                        HerNorForm[k][i] = 0
    extra_zeros = np.zeros((q.shape[0], q.shape[1] - HerNorForm.shape[1]), dtype=int)
    full_hermite = np.concatenate((HerNorForm, extra_zeros), axis=1)
    return HerNorForm


def generate_random_matrix(n, low=-3, high=3):
    A = np.random.randint(low, high + 1, size=(n, n))
    while np.linalg.matrix_rank(A[:, :n]) < n:
        A = np.random.randint(low, high + 1, size=(n, n))
    return A


def benchmark(func, matrix):
    start = time.perf_counter()
    _ = func(matrix)
    end = time.perf_counter()
    return end - start


def run_tests(algorithm, sizes, generator, repeats=3):
    results = []
    for n in sizes:
        times = []
        failures = 0
        for _ in range(repeats):
            matrix = generator(n)
            try:
                t = benchmark(algorithm, matrix)
            except Exception as e:
                print(f"Error for size {n}: {e}")
                t = float('nan')
                failures += 1
            times.append(t)
        valid_times = [t for t in times if not math.isnan(t)]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else float('nan')
        failure_rate = failures / repeats
        results.append((n, avg_time, failure_rate))
    return results


def plot_results(results, label="Hermite Normal Form Algorithm"):
    sizes = [r[0] for r in results]
    times = [r[1] for r in results]
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker='o', label=label)
    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Average runtime (seconds)")
    plt.title("Benchmark of Hermite Normal Form Algorithm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_results_to_csv(results, filename="hermite_benchmark_results(2).csv"):
    save_dir = r"C:\Users\lieke\Documents\Universiteit\Bachelor Assignment\Plotjes"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Matrix size (n x n)", "Avg runtime (s)", "Failure rate (%)"])
        for size, avg_time, failure_rate in results:
            writer.writerow([f"{size}x{2*size}", f"{avg_time:.6f}", f"{failure_rate * 100:.1f}"])
    print(f"CSV successfully saved to: {filepath}")


if __name__ == "__main__":
    sizes = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    results = run_tests(hermite_algorithm, sizes, generate_random_matrix, repeats=3)
    for n, avg, fail in results:
        print(f"Size {n}x{n}:\t{avg:.6f} s\tFailures: {fail*100:.1f}%")
    plot_results(results)
    save_results_to_csv(results)
