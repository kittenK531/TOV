from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np


def plot_pic(
    x,
    y,
    z,
    label_1,
    label_2,
    title_1,
    title_2,
    filname_1,
    filname_2,
    foldername,
    xlabel_1,
    ylabel_1,
    xlabel_2,
    ylabel_2,
):

    plt.plot(x / 1000, y, label=rf"{label_1}", color="blue")
    plt.title(rf"{title_1}")
    plt.xlabel(rf"{xlabel_1}")
    plt.ylabel(rf"{ylabel_1}")
    plt.savefig(PurePath().joinpath(foldername, filname_1))
    plt.show(block=False)
    plt.close()

    plt.plot(x / 1000, z / 1.989e30, label=rf"{label_2}", color="blue")
    plt.title(rf"{title_2}")
    plt.xlabel(rf"{xlabel_2}")
    plt.ylabel(rf"{ylabel_2}")
    plt.savefig(PurePath().joinpath(foldername, filname_2))
    plt.show(block=False)
    plt.close()


class associated_ODE:
    def __init__(self, f, g, x0, y0, z0, a, b, N):
        self.f = f
        self.g = g
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.a = a
        self.b = b
        self.N = N
        self.h = (b - a) / N

    def k0(self, x, y, z):
        return self.h * self.f(x, y, z)

    def l0(self, x, y, z):
        return self.h * self.g(x, y, z)

    def k1(self, x, y, z):
        return self.h * self.f(
            x + 0.5 * self.h, y + 0.5 * self.k0(x, y, z), z + 0.5 * self.l0(x, y, z)
        )

    def l1(self, x, y, z):
        return self.h * self.g(
            x + 0.5 * self.h, y + 0.5 * self.k0(x, y, z), z + 0.5 * self.l0(x, y, z)
        )

    def k2(self, x, y, z):
        return self.h * self.f(
            x + 0.5 * self.h, y + 0.5 * self.k1(x, y, z), z + 0.5 * self.l1(x, y, z)
        )

    def l2(self, x, y, z):
        return self.h * self.g(
            x + 0.5 * self.h, y + 0.5 * self.k1(x, y, z), z + 0.5 * self.l1(x, y, z)
        )

    def k3(self, x, y, z):
        return self.h * self.f(x + self.h, y + self.k2(x, y, z), z + self.l2(x, y, z))

    def l3(self, x, y, z):
        return self.h * self.g(x + self.h, y + self.k2(x, y, z), z + self.l2(x, y, z))

    def RK4(self):

        x, y, z = np.zeros(self.N), np.zeros(self.N), np.zeros(self.N)
        y[0], z[0] = self.y0, self.z0

        x[0] = self.a

        for i in range(self.N - 1):

            y[i + 1] = (
                y[i]
                + (
                    self.k0(x[i], y[i], z[i])
                    + 2 * self.k1(x[i], y[i], z[i])
                    + 2 * self.k2(x[i], y[i], z[i])
                    + self.k3(x[i], y[i], z[i])
                )
                / 6
            )
            z[i + 1] = (
                z[i]
                + (
                    self.l0(x[i], y[i], z[i])
                    + 2 * self.l1(x[i], y[i], z[i])
                    + 2 * self.l2(x[i], y[i], z[i])
                    + self.l3(x[i], y[i], z[i])
                )
                / 6
            )

            x[i + 1] = x[i] + self.h

        # print(x, y, z)
        print(f"radius: {x / 1000}\npressure: {y}\nmass: {z}")

        plot_pic(
            x,
            y,
            z,
            "$P(r)$",
            "$m(r)$",
            "pressure $p$ against $r$",
            "mass $m$ against $r$",
            "p(r).png",
            "m(r).png",
            "result",
        )


""" ode RK4"""


def ode_RK4(f, X_0, a, dt, T):
    N_t = int(round((T - a) / dt))
    # Initial conditions
    usol = X_0
    u = np.copy(X_0)

    tt = np.linspace(a, N_t * dt, N_t + 1)
    # RK4
    for t in tt[:-1]:
        u1 = f(u + 0.5 * dt * f(u, t), t + 0.5 * dt)
        u2 = f(u + 0.5 * dt * u1, t + 0.5 * dt)
        u3 = f(u + dt * u2, t + dt)
        u = u + (1 / 6) * dt * (f(u, t) + 2 * u1 + 2 * u2 + u3)
        usol = np.vstack((usol, u))
    return usol, tt
