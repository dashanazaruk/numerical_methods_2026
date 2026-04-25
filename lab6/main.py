import random
import math

def generate_matrix(n):
    A = []
    for i in range(n):
        row=[]
        for j in range(n):
            row.append(random.randint(1,20))
        A.append(row)
    return A

def write_matrix(filename, A):
    with open(filename, "w") as f:
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")

def read_matrix(filename):
    A=[]
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

def lu_decomposition(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    U = [[0]*n for _ in range(n)]

    for i in range(n):
        U[i][i]=1

    for k in range(n):

        for i in range(k,n):
            s = 0
            for j in range(k):
                s += L[i][j]*U[j][k]

            L[i][k]= A[i][k] - s

        for i in range(k+1,n):
            s = 0
            for j in range(k):
                s += L[k][j]*U[j][i]

            U[k][i] = (A[k][i] - s)/L[k][k]
    return L, U

def write_lu_decomposition(filename, L, U):
    with open(filename, "w") as f:
        f.write("Matrix L:\n")
        for row in L:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\nMatrix U:\n")
        for row in U:
            f.write(" ".join(map(str, row)) + "\n")

def forward_substitution(L, b):
    n = len(L)
    z =[0]*n

    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j]*z[j]
        z[i] = (b[i] - s)/L[i][i]
    return z

def backward_substitution(U, z):
    n = len(U)
    x = [0]*n

    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += U[i][j]*x[j]
        x[i] = z[i] - s
    return x

def solve_lu_decomposition(L, U, b):
    z = forward_substitution(L, b)
    x = backward_substitution(U, z)
    return x

def calculate_error(A, x, b):
    n =len(A)
    max_error = 0

    for i in range(n):
        s = 0
        for j in range(n):
            s += A[i][j]*x[j]

        error = abs(s - b[i])
        if error > max_error:
            max_error = error
    return max_error

def iterative(A, L, U, b, x0, eps=1e-14):
    x = x0[:]
    iterations = 0
    max_iterations = 100

    while iterations < max_iterations:
        Ax = matrix_vector_mult(A, x)
        R = [b[i] - Ax[i] for i in range(len(b))]
        dx = solve_lu_decomposition(L, U, R)
        x_new = [x[i] + dx[i] for i in range(len(x))]

        iterations += 1

        if norm_vector(dx) <= eps and norm_vector(R) <= eps:
            print("Ітераційне уточнення завершено успішно.")
            return x_new, iterations

        x = x_new

    print("Досягнуто максимальну кількість ітерацій.")
    return x, iterations
def main():
    n = 100

    A = generate_matrix(n)
    write_matrix("matrix_A.txt", A)
    print("Матрицю A записано у файл matrix_A.txt")

    x_true = [2.5]*n

    B = matrix_vector_mult(A, x_true)
    write_vector("vector_B.txt", B)
    print("Вектор B записано у файл vector_B.txt")

    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")

    L, U = lu_decomposition(A)
    write_lu_decomposition("LU_result.txt", L, U)
    print("LU-розклад записано у файл LU_result.txt")

    print("Розв'язок системи:")
    x = solve_lu_decomposition(L, U, B)

    print("Перші 10 значень розв'язку:")
    for i in range(10):
        print(f"x[{i + 1}] = {x[i]}")

    print("Оцінка точності:")
    error = calculate_error(A, x, B)
    print("eps =", error)

    print("Ітераційне уточнення:")
    refined_x, iterations = iterative(A, L, U, B, x)

    print("Кількість ітерацій:", iterations)

    print("Перші 10 значень уточненого розв'язку:")
    for i in range(10):
        print(f"x_refined[{i + 1}] = {refined_x[i]}")

main()
