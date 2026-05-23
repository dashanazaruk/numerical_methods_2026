import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x + y

def exact(x):
    return 2 * np.exp(x) - x - 1

x0, x_end = 0.0, 1.0
y0 = 1.0
h = 0.1
eps = 1e-4

def rk4_step(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    k3 = h * f(x + h/2, y + k2/2)
    k4 = h * f(x + h,   y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def rk4(x0, x_end, y0, h):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    while x + h <= x_end + 1e-12:
        y = rk4_step(x, y, h)
        x += h
        xs.append(round(x, 10))
        ys.append(y)
    return np.array(xs), np.array(ys)

def rk4_runge_error(h):
    xs1, ys1 = rk4(x0, x_end, y0, h)
    xs2, ys2 = rk4(x0, x_end, y0, h/2)
    ys2_c = ys2[::2]
    n = min(len(ys1), len(ys2_c))
    return xs1[:n], np.abs(ys1[:n] - ys2_c[:n]) / (2**4 - 1)

def rk4_adaptive(x0, x_end, y0, eps):
    h = (x_end - x0) / 10
    x, y = x0, y0
    xs, ys, hs = [x], [y], []
    while x < x_end - 1e-12:
        h = min(h, x_end - x)
        y1 = rk4_step(x, y, h)
        ya = rk4_step(x,     y,  h/2)
        y2 = rk4_step(x+h/2, ya, h/2)
        err = abs(y1 - y2) / (2**4 - 1)
        if err <= eps:
            x += h; y = y2
            xs.append(x); ys.append(y); hs.append(h)
            if err < eps / 32:
                h *= 2
        else:
            h /= 2
    return np.array(xs), np.array(ys), np.array(hs)

def adams2(x0, x_end, y0, h):
    xs = [x0, x0 + h]
    ys = [y0, rk4_step(x0, y0, h)]
    x = xs[-1]
    while x + h <= x_end + 1e-12:
        fn   = f(xs[-1], ys[-1])
        fn_1 = f(xs[-2], ys[-2])
        y_pred = ys[-1] + h/2 * (3*fn - fn_1)
        x_new  = xs[-1] + h

        y_corr = y_pred
        for _ in range(2):
            y_corr = ys[-1] + h/2 * (f(x_new, y_corr) + fn)
        xs.append(round(x_new, 10))
        ys.append(y_corr)
        x = x_new
    return np.array(xs), np.array(ys)

def adams2_runge_error(h):
    xs1, ys1 = adams2(x0, x_end, y0, h)
    xs2, ys2 = adams2(x0, x_end, y0, h/2)
    ys2_c = ys2[::2]
    n = min(len(ys1), len(ys2_c))
    return xs1[:n], np.abs(ys1[:n] - ys2_c[:n]) / (2**2 - 1)

def adams2_adaptive(x0, x_end, y0, eps):
    h = (x_end - x0) / 10

    xs = [x0, x0 + h]
    ys = [y0, rk4_step(x0, y0, h)]
    hs = []
    x = xs[-1]
    while x < x_end - 1e-12:
        h = min(h, x_end - x)
        fn   = f(xs[-1], ys[-1])
        fn_1 = f(xs[-2], ys[-2])

        yp = ys[-1] + h/2 * (3*fn - fn_1)
        y1 = ys[-1] + h/2 * (f(xs[-1]+h, yp) + fn)

        yp2 = ys[-1] + (h/2)/2 * (3*fn - fn_1)
        ym  = ys[-1] + (h/2)/2 * (f(xs[-1]+h/2, yp2) + fn)
        fn_m = f(xs[-1]+h/2, ym)
        yp3 = ym + (h/2)/2 * (3*fn_m - fn)
        y2  = ym + (h/2)/2 * (f(xs[-1]+h, yp3) + fn_m)

        err = abs(y1 - y2) / (2**2 - 1)
        if err <= eps:
            xs.append(xs[-1] + h); ys.append(y2); hs.append(h)
            x = xs[-1]
            if err < eps / 8:
                h *= 2
        else:
            h /= 2
    return np.array(xs), np.array(ys), np.array(hs)

def print_table(xs, ys, label):
    ye = exact(xs)
    print(f"\n{label}")
    print(f"{'x':>6} | {'y_числ':>14} | {'y_точн':>14} | {'|похибка|':>12}")
    print("-" * 52)
    for x, y, ye_ in zip(xs, ys, ye):
        print(f"{x:>6.3f} | {y:>14.8f} | {ye_:>14.8f} | {abs(y-ye_):>12.2e}")
    print(f"  Макс. похибка: {np.max(np.abs(ys - ye)):.4e}")

xs_ad, ys_ad = adams2(x0, x_end, y0, h)
print_table(xs_ad, ys_ad, f"Адамс PC2, h={h}")

xs_ar, err_ar = adams2_runge_error(h)
print(f"\n  Оцінка похибки (Рунге), макс: {err_ar.max():.4e}")
print(f"  Крок h={h} {'оптимальний' if err_ar.max() < eps else 'потребує зменшення'} (ε={eps})")

xs_aa, ys_aa, hs_aa = adams2_adaptive(x0, x_end, y0, eps)
print(f"\n  Адаптивний крок Адамса: {len(xs_aa)} вузлів, h від {hs_aa.min():.4f} до {hs_aa.max():.4f}")

# Ч.2 — Рунге-Кутта 4
xs_rk, ys_rk = rk4(x0, x_end, y0, h)
print_table(xs_rk, ys_rk, f"\nRK4, h={h}")

xs_rr, err_rr = rk4_runge_error(h)
print(f"\n  Оцінка похибки RK4 (Рунге), макс: {err_rr.max():.4e}")
h_opt = h * (eps / err_rr.max()) ** (1/4)
print(f"  Оціночний оптимальний крок для ε={eps}: h ≈ {h_opt:.5f}")

xs_ra, ys_ra, hs_ra = rk4_adaptive(x0, x_end, y0, eps)
print(f"\n  Адаптивний крок RK4: {len(xs_ra)} вузлів, h від {hs_ra.min():.4f} до {hs_ra.max():.4f}")

x_plot = np.linspace(x0, x_end, 500)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Ч.1 — Метод Адамса прогнозу-корекції 2-го порядку", fontsize=13, fontweight='bold')

axes[0,0].plot(x_plot, exact(x_plot), 'b-', label="Точний")
axes[0,0].plot(xs_ad, ys_ad, 'r--o', ms=5, label=f"Адамс (h={h})")
axes[0,0].set_title("Розв'язок ОДУ"); axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].semilogy(xs_ad, np.abs(ys_ad - exact(xs_ad)) + 1e-18, 'r-o', ms=4)
axes[0,1].set_title(f"Локальна похибка (точна), h={h}")
axes[0,1].set_xlabel("x"); axes[0,1].set_ylabel("|err|"); axes[0,1].grid(True)

axes[1,0].semilogy(xs_ar, err_ar + 1e-18, 'g-s', ms=4, label="Рунге")
axes[1,0].semilogy(xs_ad, np.abs(ys_ad - exact(xs_ad)) + 1e-18, 'r--o', ms=3, label="Точна")
axes[1,0].axhline(eps, color='k', ls=':', label=f"ε={eps}")
axes[1,0].set_title("Оцінка похибки методом Рунге"); axes[1,0].legend(); axes[1,0].grid(True)

axes[1,1].step(xs_aa[:len(hs_aa)], hs_aa, where='post', color='purple')
axes[1,1].set_title(f"Автоматичний вибір кроку (ε={eps})")
axes[1,1].set_xlabel("x"); axes[1,1].set_ylabel("h"); axes[1,1].grid(True)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Ч.2 — Метод Рунге-Кутта 4-го порядку", fontsize=13, fontweight='bold')

axes[0,0].plot(x_plot, exact(x_plot), 'b-', label="Точний")
axes[0,0].plot(xs_rk, ys_rk, 'r--o', ms=5, label=f"RK4 (h={h})")
axes[0,0].set_title("Розв'язок ОДУ"); axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].semilogy(xs_rk, np.abs(ys_rk - exact(xs_rk)) + 1e-18, 'r-o', ms=4)
axes[0,1].set_title(f"Локальна похибка (точна), h={h}")
axes[0,1].set_xlabel("x"); axes[0,1].set_ylabel("|err|"); axes[0,1].grid(True)

axes[1,0].semilogy(xs_rr, err_rr + 1e-18, 'g-s', ms=4, label="Рунге")
axes[1,0].semilogy(xs_rk, np.abs(ys_rk - exact(xs_rk)) + 1e-18, 'r--o', ms=3, label="Точна")
axes[1,0].axhline(eps, color='k', ls=':', label=f"ε={eps}")
axes[1,0].set_title("Оцінка похибки методом Рунге"); axes[1,0].legend(); axes[1,0].grid(True)

axes[1,1].step(xs_ra[:len(hs_ra)], hs_ra, where='post', color='purple')
axes[1,1].set_title(f"Автоматичний вибір кроку (ε={eps})")
axes[1,1].set_xlabel("x"); axes[1,1].set_ylabel("h"); axes[1,1].grid(True)

plt.tight_layout()
plt.show()