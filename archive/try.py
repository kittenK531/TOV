import pylab
from scipy.constants import G, c, hbar, m_n, pi
from scipy.integrate import odeint

Msun = 1.98892e30

# piecewise polytrope equation of state
Gamma0 = 5.0 / 3.0  # low densities: soft non−relativistic degeneracy pressure
K0 = (3.0 * pi ** 2) ** (2.0 / 3.0) * hbar ** 2 / (5.0 * m_n ** (8.0 / 3.0))
Gamma1 = 3.0  # high densities: stiffer equation of state
rho1 = 5e17
P1 = K0 * rho1 ** Gamma0
K1 = P1 / rho1 ** Gamma1


def eos(rho):
    if rho < rho1:
        return K0 * rho ** Gamma0
    else:
        return K1 * rho ** Gamma1


def inveos(P):
    if P < P1:
        return (P / K0) ** (1.0 / Gamma0)
    else:
        return (P / K1) ** (1.0 / Gamma1)


def tov(y, r):
    P, m = y[0], y[1]
    rho = inveos(P)
    dPdr = -1 * G * (rho + P / c ** 2) * (m + 4.0 * pi * r ** 3 * P / c ** 2)
    dPdr = dPdr / (r * (r - 2 * G * m / c ** 2))
    dmdr = 4.0 * pi * r ** 2 * rho * (1 + P / ((2.75 - 1) * rho) / c ** 2)
    return [dPdr, dmdr]


def tovsolve(rhoc):
    r = pylab.arange(10.0, 20000.0, 10)
    m = pylab.zeros_like(r)
    P = pylab.zeros_like(r)
    P[0] = eos(rhoc)
    m[0] = 4.0 * pi * r[0] ** 3 * rhoc
    y0 = pylab.array([P[0], m[0]])

    y = odeint(tov, y0, r)

    # print(y)

    return y


rhoc = pylab.logspace(17.5, 20)

print(rhoc)

for rho_c in rhoc:
    sol = tovsolve(rho_c)

    R = pylab.arange(10.0, 20000.0, 10)
    P = sol[:, 0]
    M = sol[:, 1]

    pylab.plot(R / 1000, M, label=f"$\rho$ {rho_c}")

pylab.xlabel("Radius (km)")
pylab.ylabel("Mass (solar)")
pylab.show()

for rho_c in rhoc:

    print(rho_c)
    sol = tovsolve(rho_c)

    R = pylab.arange(10.0, 20000.0, 10)
    P = sol[:, 0]
    M = sol[:, 1]

    pylab.plot(R / 1000, P)
pylab.xlabel("Radius (km)")
pylab.ylabel("Pressure")
pylab.show()
