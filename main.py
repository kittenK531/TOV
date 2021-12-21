import numpy as np

from rk4 import ode_RK4, plot_pic

""" Universal constant """
M_0 = 1.989e30  # (kg)
G = 6.67e-11
c = 3e8

""" Defining variables for the polytropic equation of state """
gamma = 2.75
K = 1.11456e-15  # (SI unit)
const = (K ** (1 / gamma)) / (gamma - 1)
const2 = gamma / (gamma - 1)

""" Define object """
R_c = 8000  # (m)
M_c = 1.4 * M_0  # (kg)
rho_c = 5e17  # (SI unit)
P_c = K * rho_c ** gamma

P1 = K * rho_c ** gamma
K1 = P1 / rho_c ** gamma

"""" Polytropic equation of state """


def eos(rho_i):
    return K * rho_i ** gamma


def rho(p):
    return (p / K) ** (1.0 / gamma)


def epsilon(p):
    # return p ** (1 - 1 / gamma) * const
    return p / ((gamma - 1) * rho(p))


def dpdr(r, p, m):

    dPdr = (
        -1
        * G
        * (rho(p) + p / c ** 2 * const2)
        * (m + 4.0 * np.pi * r ** 3 * p / c ** 2)
    )
    dPdr = dPdr / (r * (r - 2 * G * m / c ** 2))

    return dPdr


def dmdr(r, p, m):
    return 4 * np.pi * r ** 2 * rho(p) * (1 + p / ((gamma - 1) * rho(p)) / c ** 2)


def tov(y, r):

    p, m = y

    dp = dpdr(r, p, m)
    dm = dmdr(r, p, m)

    return np.array([dp, dm])


def tovsolve(rho_i):

    a = 10
    b = 11000
    dr = 10

    r = np.arange(a, b, dr)

    y0 = np.array([eos(rho_i), 4.0 * np.pi * r[0] ** 3 * rho_i])  # TODO

    y, r = ode_RK4(tov, y0, a, dr, b)

    return y, r


sol, r = tovsolve(rho_c)

P = sol[:, 0]
M = sol[:, 1]

plot_pic(
    r,
    P,
    M,
    "$P(r)$",
    "$m(r)$",
    "pressure $p$ against $r$",
    "mass $m$ against $r$",
    "p(r).png",
    "m(r).png",
    "result",
    "radius (km)",
    "pressure (Pa)",
    "radius (km)",
    "mass $(M_\odot)$",
)
