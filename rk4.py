import numpy as np


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

        print(x, y, z)
