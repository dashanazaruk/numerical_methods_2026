import csv
import numpy as np
import matplotlib.pyplot as plt
import math

def create_csv():
    data = [
        ["Dataset size", "Train time (sec)"],
        [10000, 8],
        [20000, 20],
        [40000, 55],
        [80000, 150],
        [160000, 420]
    ]
    with open("data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print("CSV файл створено")

def read_csv(filename):
    x = []
    y = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(x), np.array(y)
#фунція для таблиці розділення різниць
def divided_diff(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef
#функція для обчислення многочлена Ньютона в точці
def newton_polynomial(coef, x_data, x):
    n = len(coef)
    p = coef[n-1]
    for k in range(1, n):
        p = coef[-k] + (x - x_data[-k]) * p
    return p
def factorial_polynomial(x_data, y_data, x):
    h = x_data[1] - x_data[0]
    diff = [y_data.copy()]
    for i in range(1, len(y_data)):
        diff.append(np.diff(diff[i-1]))
    t = (x - x_data[0]) / h
    result = y_data[0]
    for i in range(1, len(diff)):
        term = diff[i][0]
        for j in range(i):
            term *= (t - j)
        term /= math.factorial(i)
        result += term
    return result

def main():
    create_csv()
    x, y = read_csv("data.csv")
    coef = divided_diff(x, y)

    prediction_newton = newton_polynomial(coef, x, 120000)
    prediction_factorial = factorial_polynomial(x, y, 120000)

    print("Прогноз для 120000 (Ньютон):", prediction_newton)
    print("Прогноз для 120000 (Факторіальний):", prediction_factorial)

    x_plot = np.linspace(min(x), max(x), 200)
    y_newton_plot = [newton_polynomial(coef, x, xi) for xi in x_plot]
    y_factorial_plot = [factorial_polynomial(x, y, xi) for xi in x_plot]

    plt.figure(figsize=(8,5))

    plt.scatter(x, y, color='black', label="Вузли")
    plt.plot(x_plot, y_newton_plot, label="Ньютон")
    plt.plot(x_plot, y_factorial_plot, label="Факторіальний")

    plt.title("Інтерполяційна модель")
    plt.xlabel("Dataset size")
    plt.ylabel("Train time (sec)")
    plt.legend()
    plt.grid(True)
    plt.show()

    nodes_list = [5, 10, 20]
    plt.figure(figsize=(8,5))
    y_true_interp = np.interp(x_plot, x, y)

    for n in nodes_list:
        x_sub = x[:min(n, len(x))]
        y_sub = y[:min(n, len(y))]

        coef_sub = divided_diff(x_sub, y_sub)
        y_sub_plot = np.array([newton_polynomial(coef_sub, x_sub, xi) for xi in x_plot])
        error = np.abs(y_true_interp - y_sub_plot)
        plt.plot(x_plot, error, label=f"n={n}")

    plt.title("Графік похибок інтерполяції (ефект Рунге)")
    plt.xlabel("Dataset size")
    plt.ylabel("Абсолютна похибка")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,5))

    for n in nodes_list:
        x_sub = x[:min(n, len(x))]
        y_sub = y[:min(n, len(y))]
        coef_sub = divided_diff(x_sub, y_sub)
        y_sub_plot = np.array([newton_polynomial(coef_sub, x_sub, xi) for xi in x_plot])
        fluctuation = np.abs(np.diff(y_sub_plot))
        plt.plot(x_plot[1:], fluctuation, label=f"n={n}")

    plt.title("Коливання полінома Ньютона")
    plt.xlabel("Dataset size")
    plt.ylabel("Δ значення полінома")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()