import numpy as np

from rk4 import associated_ODE

""" Defining variables for the polytropic equation of state """
K = 1.11456e-15
gamma = 2.75
G = 6.67e-11
c = 3e8

""" Define object """
rho_c = 5.0e17  # (SI unit)
P_c = K * rho_c ** gamma

"""" Polytropic equation of state """


class tov:
    def __init__(self) -> None:
        pass

    def epsilon(self, p):
        return (p ** (1 - 1 / gamma) * (K ** (1 / gamma))) / (gamma - 1)

    def rho(self, p):
        return (p / K) ** (1 / gamma)

    def dpdr(self, r, p, m):
        return (
            -1
            * G
            * (self.rho(p) * (1 + self.epsilon(p) / c ** 2) + p / c ** 2)
            * ((m + 4 * np.pi * r ** 3 * p / c ** 2) / (r * (r - 2 * G * m / c ** 2)))
        )

    def dmdr(self, r, p, m):
        return 4 * np.pi * r ** 2 * self.rho(p) * (1 + self.epsilon(p) / c ** 2)


tov = tov()

soln = associated_ODE(tov.dpdr, tov.dmdr, 1, P_c, rho_c, 1, 1.5e11, 10)

soln.RK4()
