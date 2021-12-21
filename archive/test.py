import numpy as np


def ode_RK4(f, X_0, dt, T):
    N_t = int(round(T / dt))
    # Initial conditions
    usol = X_0
    u = np.copy(X_0)

    tt = np.linspace(0, N_t * dt, N_t + 1)
    # RK4
    for t in tt[:-1]:
        u1 = f(u + 0.5 * dt * f(u, t), t + 0.5 * dt)
        u2 = f(u + 0.5 * dt * u1, t + 0.5 * dt)
        u3 = f(u + dt * u2, t + dt)
        u = u + (1 / 6) * dt * (f(u, t) + 2 * u1 + 2 * u2 + u3)
        usol = np.vstack((usol, u))
    return usol, tt


def demo_exp():
    import matplotlib.pyplot as plt

    def f(u, t):
        return np.asarray([u])

    u, t = ode_RK4(f, np.array([1]), 0.1, 1.5)

    plt.plot(t, u, "b*", t, np.exp(t), "r-")
    plt.show()


def demo_osci():
    import matplotlib.pyplot as plt

    def f(u, t, omega=2):
        u, v = u
        return np.asarray([v, -(omega ** 2) * u])

    u, t = ode_RK4(f, np.array([2, 0]), 0.1, 2)

    u1 = [a[0] for a in u]

    for i in [1]:
        plt.plot(t, u1, "b*")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

a = 1
b = 1
c = 1
d = 1


# du/dt=V(u,t)
def V(u, t):
    x, y, vx, vy = u
    return np.array([vy, vx, a * x + b * y, c * x + d * y])


def rk4(f, u0, t0, tf, n):
    t = np.linspace(t0, tf, n + 1)
    u = np.array((n + 1) * [u0])
    h = t[1] - t[0]
    for i in range(n):
        k1 = h * f(u[i], t[i])
        k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(u[i] + k3, t[i] + h)
        u[i + 1] = u[i] + (k1 + 2 * (k2 + k3) + k4) / 6
    return u, t


u, t = rk4(V, np.array([1.0, 0.0, 1.0, 0.0]), 0.0, 10.0, 100000)
x, y, vx, vy = u.T
# plt.plot(t, x, t,y)
plt.semilogy(t, x, t, y)
plt.grid("on")
plt.show()
