import random

def generate_matrix(n):
    A = []

    for i in range(n):
        row = []

        for j in range(n):
            if i != j:
                row.append(random.randint(1, 20))
            else:
                row.append(0)

        row_sum = sum(abs(x) for x in row)
        row[i] = row_sum + random.randint(1, 20)

        A.append(row)

    return A

def write_matrix(filename, A):
    with open(filename, "w") as f:
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")

def read_matrix(filename):
    A = []

    with open(filename, "r") as f:
        for line in f:
            A.append(list(map(float, line.split())))

    return A

def write_vector(filename, v):
    with open(filename, "w") as f:
        for x in v:
            f.write(str(x) + "\n")


def read_vector(filename):
    v = []

    with open(filename, "r") as f:
        for line in f:
            v.append(float(line.strip()))

    return v

def matrix_vector_mult(A, x):
    n = len(A)
    result = [0] * n

    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * x[j]

    return result

def norm_vector(v):
    return max(abs(x) for x in v)

def norm_matrix(A):
    n = len(A)
    return max(sum(abs(A[i][j]) for j in range(n)) for i in range(n))

def simple_iteration(A, b, x0, eps=1e-14, max_iter=10000):
    n = len(A)
    x = x0[:]

    tau = 1 / norm_matrix(A)

    for k in range(max_iter):
        Ax = matrix_vector_mult(A, x)

        x_new = [0] * n

        for i in range(n):
            x_new[i] = x[i] - tau * (Ax[i] - b[i])

        diff = [x_new[i] - x[i] for i in range(n)]

        if norm_vector(diff) < eps:
            return x_new, k + 1

        x = x_new

    return x, max_iter

def jacobi_method(A, b, x0, eps=1e-14, max_iter=10000):
    n = len(A)
    x = x0[:]

    for k in range(max_iter):
        x_new = [0] * n

        for i in range(n):
            s = 0

            for j in range(n):
                if i != j:
                    s += A[i][j] * x[j]

            x_new[i] = (b[i] - s) / A[i][i]

        diff = [x_new[i] - x[i] for i in range(n)]

        if norm_vector(diff) < eps:
            return x_new, k + 1

        x = x_new

    return x, max_iter

def seidel_method(A, b, x0, eps=1e-14, max_iter=10000):
    n = len(A)
    x = x0[:]

    for k in range(max_iter):
        x_old = x[:]

        for i in range(n):
            s1 = 0
            s2 = 0

            for j in range(i):
                s1 += A[i][j] * x[j]

            for j in range(i + 1, n):
                s2 += A[i][j] * x_old[j]

            x[i] = (b[i] - s1 - s2) / A[i][i]

        diff = [x[i] - x_old[i] for i in range(n)]

        if norm_vector(diff) < eps:
            return x, k + 1

    return x, max_iter

def main():
    n = 100

    A = generate_matrix(n)
    write_matrix("matrix_A.txt", A)
    print("Матрицю A записано у файл matrix_A.txt")

    x_true = [2.5] * n

    B = matrix_vector_mult(A, x_true)
    write_vector("vector_B.txt", B)
    print("Вектор B записано у файл vector_B.txt")

    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")

    x0 = [1.0] * n
    eps = 1e-14

    print("\nМетод простої ітерації:")
    x1, it1 = simple_iteration(A, B, x0, eps)
    print("Кількість ітерацій:", it1)
    print("Перші 10 значень:", x1[:10])

    print("\nМетод Якобі:")
    x2, it2 = jacobi_method(A, B, x0, eps)
    print("Кількість ітерацій:", it2)
    print("Перші 10 значень:", x2[:10])

    print("\nМетод Зейделя:")
    x3, it3 = seidel_method(A, B, x0, eps)
    print("Кількість ітерацій:", it3)
    print("Перші 10 значень:", x3[:10])

    print("\nТочний розв'язок:")
    print(x_true[:10])


main()