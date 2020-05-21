
import numpy as np
from constants import Constants

def Gibbs(x, p):
    '''
    Arguments:
    T -- input temperature
    P -- input pressure
    x -- list of moles of compound in the following order CH4, CO2, H2O, O2, CO, H2, He, C

    Returns:
    Gibbs free energy
    '''
    # converts the list of moles of each compound into an numpy array
    nj = np.array([x])

    # Defines the T and p, from the parameters list
    T = p[0]
    P = p[1]

    # Vector to find the Delta free energy at a specific temperature
    T_vector = np.array([[1], [T], [T**2], [T**3], [T**4]])

    Enj = np.sum(nj) # total moles
    y_j = nj / Enj

    # vector for the coefficients of fugacity
    phi = np.ones((1, 8))

    # Gibbs free energy of formation, J/mol
    Gj0 = 1000 * np.dot(Constants.DelG_f, T_vector)

    first = np.sum(nj[0][0:-1] * Gj0.T[0][0:-1])
    second = Constants.R * T * np.sum(nj[0][0:-1] * np.log(y_j[0][0:-1] * phi[0][0:-1] * P / Constants.P0))
    third = nj[0][-1] * Gj0.T[0][-1]

    G = first + second + third

    return G

# nj = [0.3, 0.1, 0.5, 0.01, 0.5, 0.5, 0.5, 0.5]

# print(Gibbs(nj, np.array([1000, 1])))
