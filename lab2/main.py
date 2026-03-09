import csv
import numpy as np
import matplotlib.pyplot as plt

def create_csv():
    data = [
        ["x", "y"],
        [10000, 2],
        [20000, 3],
        [40000, 5],
        [60000, 7],
        [80000, 9],
        [100000, 12]
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

def divided_diff(x, y):
    n = len(y)
    coef = np.copy(y)

    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])

    return coef

def newton_poly(coef, x_data, x):
    n = len(coef)
    p = coef[n-1]

    for k in range(1, n):
        p = coef[n-k-1] + (x - x_data[n-k-1]) * p

    return p

def main():

    create_csv()

    x, y = read_csv("data.csv")

    coef = divided_diff(x, y)

    prediction = newton_poly(coef, x, 120000)
    print("Прогноз для 120000:", prediction)

    x_plot = np.linspace(min(x), max(x), 200)
    y_plot = [newton_poly(coef, x, i) for i in x_plot]

    plt.scatter(x, y, label="Дані")
    plt.plot(x_plot, y_plot, label="Інтерполяція Ньютона")
    plt.legend()
    plt.title("Інтерполяційна модель")
    plt.show()

    nodes = [5, 10, 20]

    for n in nodes:
        x_sub = x[:min(n, len(x))]
        y_sub = y[:min(n, len(y))]

        coef_sub = divided_diff(x_sub, y_sub)
        y_sub_plot = [newton_poly(coef_sub, x_sub, i) for i in x_plot]

        plt.plot(x_plot, y_sub_plot, label=f"n={n}")

    plt.scatter(x, y)
    plt.legend()
    plt.title("Ефект Рунге")
    plt.show()


if __name__ == "__main__":
    main()