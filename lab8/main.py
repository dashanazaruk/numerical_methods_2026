import math
import numpy as np
import matplotlib.pyplot as plt

def F(x):
    return x**3 - 2*x + 1

def dF(x):
    return 3*x**2 - 2

def ddF(x):
    return 6*x

def tabulate_function(a, b, h, filename):
    points = []
    with open(filename, "w") as file:
        x = a
        while x <= b:

            y = F(x)
            points.append((x, y))
            file.write(f"x = {x:.4f}    y = {y:.10f}\n")
            x = round(x + h, 10)

    return points

def plot_function(a, b):

    x = np.linspace(a, b, 1000)
    y = F(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="F(x) = x^3 - 4x + 1")
    plt.axhline(0)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Графік функції")
    plt.legend()
    plt.show()

def find_intervals(points):

    intervals = []

    for i in range(len(points) - 1):

        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        if y1 * y2 < 0:
            intervals.append((x1, x2))

    return intervals

def simple_iteration(x0, eps, max_iter=1000):
    def phi(x):
        return x - 0.01 * F(x)

    x_prev = x0

    for i in range(max_iter):

        x_next = phi(x_prev)
        if abs(F(x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, i + 1
        x_prev = x_next

    return x_prev, max_iter

def newton_method(x0, eps, max_iter=1000):
    x = x0
    for i in range(max_iter):
        if dF(x) == 0:
            return None, i + 1
        x_next = x - F(x) / dF(x)

        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next

    return x, max_iter

def chebyshev_method(x0, eps, max_iter=1000):
    x = x0

    for i in range(max_iter):
        fx = F(x)
        dfx = dF(x)
        ddfx = ddF(x)
        if dfx == 0:
            return None, i + 1
        x_next = x - fx / dfx - (ddfx * fx**2) / (2 * dfx**3)

        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next

    return x, max_iter

def chord_method(a, b, eps, max_iter=1000):
    x_prev = a

    for i in range(max_iter):

        denominator = F(b) - F(x_prev)

        if denominator == 0:
            return None, i + 1
        x_next = x_prev - F(x_prev) * (b - x_prev) / denominator

        if abs(F(x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, i + 1
        x_prev = x_next

    return x_prev, max_iter

def parabola_method(x0, eps, max_iter=1000):
    x = x0

    for i in range(max_iter):

        fx = F(x)
        dfx = dF(x)
        ddfx = ddF(x)

        D = dfx**2 - 2 * fx * ddfx

        if D < 0:
            return None, i + 1
        denominator = dfx + math.sqrt(D)

        if denominator == 0:
            return None, i + 1
        x_next = x - (2 * fx) / denominator

        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next

    return x, max_iter

def inverse_interpolation(x0, x1, eps, max_iter=1000):

    for i in range(max_iter):

        f0 = F(x0)
        f1 = F(x1)
        denominator = f1 - f0

        if denominator == 0:
            return None, i + 1
        x2 = x1 - f1 * (x1 - x0) / denominator

        if abs(F(x2)) < eps and abs(x2 - x1) < eps:
            return x2, i + 1

        x0 = x1
        x1 = x2

    return x1, max_iter

def polynomial(x, coeffs):

    result = 0

    n = len(coeffs)

    for i in range(n):
        result += coeffs[i] * x**(n - i - 1)

    return result

def plot_polynomial(coeffs):

    x = np.linspace(-5, 5, 1000)

    y = []

    for value in x:
        y.append(polynomial(value, coeffs))

    plt.figure(figsize=(10, 6))

    plt.plot(x, y, label="Поліном")

    plt.axhline(0)

    plt.grid(True)

    plt.xlabel("x")
    plt.ylabel("P(x)")

    plt.title("Графік полінома")
    plt.legend()
    plt.show()


def save_coefficients(filename, coeffs):

    with open(filename, "w") as file:

        for c in coeffs:
            file.write(str(c) + "\n")

def load_coefficients(filename):

    coeffs = []

    with open(filename, "r") as file:

        for line in file:
            coeffs.append(float(line.strip()))

    return coeffs

def horner(coeffs, x):
    result = coeffs[0]
    for i in range(1, len(coeffs)):
        result = result * x + coeffs[i]

    return result

def horner_derivative(coeffs, x):
    n = len(coeffs) - 1
    derivative_coeffs = []

    for i in range(n):
        derivative_coeffs.append(coeffs[i] * (n - i))

    return horner(derivative_coeffs, x)

def newton_polynomial(coeffs, x0, eps, max_iter=1000):

    x = x0

    for i in range(max_iter):

        fx = horner(coeffs, x)
        dfx = horner_derivative(coeffs, x)

        if dfx == 0:
            return None, i + 1

        x_next = x - fx / dfx

        if abs(x_next - x) < eps:
            return x_next, i + 1

        x = x_next

    return x, max_iter

def lin_method(coeffs):

    return np.roots(coeffs)

def main():

    print("===================================")
    print("ТАБУЛЯЦІЯ ФУНКЦІЇ")
    print("===================================")

    a = -5
    b = 5
    h = 0.1

    points = tabulate_function(a, b, h, "table.txt")

    print("Таблиця записана у файл table.txt")

    plot_function(a, b)

    intervals = find_intervals(points)

    print("\nІнтервали зміни знаку:")

    for interval in intervals:
        print(interval)

    eps = 1e-10

    for interval in intervals[:2]:
        print("")
        print("Інтервал:", interval)
        print("")

        x0 = (interval[0] + interval[1]) / 2

        root, iterations = simple_iteration(x0, eps)

        print("\nМетод простої ітерації")

        if root is not None:
            print("Корінь =", root)
            print("Ітерацій =", iterations)
        else:
            print("Метод не спрацював")

        root, iterations = newton_method(x0, eps)

        print("\nМетод Ньютона")

        if root is not None:
            print("Корінь =", root)
            print("Ітерацій =", iterations)
        else:
            print("Метод не спрацював")

        root, iterations = chebyshev_method(x0, eps)

        print("\nМетод Чебишева")

        if root is not None:
            print("Корінь =", root)
            print("Ітерацій =", iterations)
        else:
            print("Метод не спрацював")

        root, iterations = chord_method(
            interval[0],
            interval[1],
            eps
        )

        print("\nМетод хорд")

        if root is not None:
            print("Корінь =", root)
            print("Ітерацій =", iterations)
        else:
            print("Метод не спрацював")

        root, iterations = parabola_method(x0, eps)

        print("\nМетод парабол")

        if root is not None:
            print("Корінь =", root)
            print("Ітерацій =", iterations)
        else:
            print("Метод не спрацював")

        root, iterations = inverse_interpolation(
            interval[0],
            interval[1],
            eps
        )

        print("\nМетод зворотної інтерполяції")

        if root is not None:
            print("Корінь =", root)
            print("Ітерацій =", iterations)
        else:
            print("Метод не спрацював")

    print("ПОЛІНОМ 3 ПОРЯДКУ")
    print("===================================")

    coeffs = [1, -2, 4, -8]

    save_coefficients("coefficients.txt", coeffs)

    print("\nКоефіцієнти записані у файл coefficients.txt")

    loaded_coeffs = load_coefficients("coefficients.txt")

    print("\nЗчитані коефіцієнти:")
    print(loaded_coeffs)

    plot_polynomial(loaded_coeffs)

    root, iterations = newton_polynomial(
        loaded_coeffs,
        2,
        eps
    )

    print("\nДійсний корінь:")

    if root is not None:
        print(root)
        print("Кількість ітерацій =", iterations)
    else:
        print("Метод не спрацював")

    complex_roots = lin_method(loaded_coeffs)

    print("\nКомплексні корені:")

    for r in complex_roots:
        print(r)

main()