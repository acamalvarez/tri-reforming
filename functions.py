from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from constants import Constants


def Gibbs(x: List[float], p: List) -> float:
    """
    Arguments:
    -- x list of moles of compound in the following order CH4, CO2, H2O, O2, CO, H2, C
    -- T input temperature
    -- P input pressure

    Returns:
    Gibbs free energy
    """

    # converts the list of moles of each compound into an numpy array
    nj = np.array(x)

    # Defines the T and p, from the parameters list
    T = p[0]
    P = p[1]

    # Vector to find the Delta free energy at a specific temperature
    T_vector = np.array([1, T, T**2, T**3, T**4])

    Enj = sum(nj) # total moles
    y_j = nj / Enj # fractions

    # vector for the coefficients of fugacity
    phi = np.ones(7)

    # Gibbs free energy of formation, kJ/mol
    Gj0 = np.dot(Constants.del_G_f, T_vector)

    first = sum(nj[0:-1] * Gj0[0:-1])
    second = Constants.R * T * sum(nj[0:-1] * np.log(y_j[0:-1] * phi[0:-1] * P / Constants.P0))
    third = nj[-1] * Gj0[-1]

    return first + second + third


#  CH4  CO2  H2O  O2    CO   H2   He   C
def calculate_mol(mol_0, T, P):
    """
    calculates the moles at the equilibrium
    Arguments:
    -- moles_0: initial moles in this order:
        CH4, CO2, H2O, O2, CO, H2, C
    -- T: temperature, °C
    -- P: pressure, atm
    Returns:
    -- numpy array with the moles at the equilibrium of
    CH4, CO2, H2O, O2, CO, H2, C

    """

    # calculates temperature and pressure to convinient units
    T = T + 273  # temperature, K
    P = 1  # pressure, bar

    # organizes the parameters (T and P) in a numpy array
    parameters = np.array([T, P])

    # calculates the vector of elemental balances
    beq = np.dot(Constants.Aeq, np.array(mol_0))

    # Defines the equality constraints with the elemental balance
    eq_cons = {"type": "eq", "fun": lambda x: np.dot(Constants.Aeq, np.array(x)) - beq}

    # options for the solver of the optimization
    options = {"ftol": 1e-6, "disp": False, "maxiter": 1000, "eps": 1.4901161193847656e-8}

    # solve the optimization problem
    result = minimize(
        Gibbs,
        Constants.n0,
        args=parameters,
        method="SLSQP",
        bounds=Constants.bnds,
        constraints=eq_cons,
        options=options,
    )

    return result.x


def find_results(mol_0, Temp, P):
    """
    returns a data frame with the results at each temperature of the
    Temp array in the following order:
    moles of each compound; conversions of CH4, CO2, and H2O; and H2/CO ratio

    Arguments:
    -- mol_0: list of initial moles in this order [CH4, CO2, H2O, O2]
    -- Temp: array of the temperature to evaluate, C
    -- P: pressure of the system, bar

    Returns:
    -- data frame of the results

    """

    # array for storing the moles
    mol = np.zeros((len(Temp), 7))

    # array for storing the conversions
    conversions = np.zeros((len(Temp), 7))

    # for loop to calculate moles and conversions at each temperature
    for i in np.arange(len(Temp)):
        # calculates moles and storages in the moles array
        mol[i] = calculate_mol(mol_0, Temp[i], P)

        # calculates conversions and storates in the conversions array
        conversions[i] = 100 * (mol_0 - mol[i]) / mol_0

    # creates moles data frame
    df_mol = pd.DataFrame(data=mol, index=Temp, columns=Constants.columns_mol)

    # creates conversions data frame
    df_conv = pd.DataFrame(data=conversions, index=Temp, columns=Constants.columns_conversions)

    # concatenate the data frames in results data frame
    df_results = pd.concat([df_mol, df_conv[["conv_CH4", "conv_CO2", "conv_H2O"]]], axis=1)

    # calculates H2/CO ratio and adds a column to the data frame results
    df_results["H2/CO"] = df_results["mol_H2"] / df_results["mol_CO"]

    return df_results


def generate_data_for_NN(mol_0, Temp, P):
    """
    Generate data that could be used when training a neural network

    Arguments:
    -- mol_0: list of initial moles in this order: [CH4, CO2, H2O, O2]
    -- Temp: array of the temperature to evaluate, C
    -- P: pressure of the system, bar

    Returns:
    Data frame with the following values: initial mol of CH4, CO2, H2O, and O2; conversion of CH4, CO2, H2O; and H2/CO ratio
    """

    finalResults = find_results(mol_0, Temp, P)[["conv_CH4", "conv_CO2", "conv_H2O", "H2/CO"]]
    finalResults["CH4_0"] = mol_0[0]
    finalResults["CO2_0"] = mol_0[1]
    finalResults["H2O_0"] = mol_0[2]
    finalResults["O2_0"] = mol_0[3]
    finalResults["Temperature"] = Temp

    return finalResults[["CH4_0", "CO2_0", "H2O_0", "O2_0", "conv_CH4", "conv_CO2", "conv_H2O", "H2/CO"]]


def csv_for_NN(moles_0, Temp, P):
    """
    Creates a csv file at each initial mol of the moles_0 array for all the temperature in the array Temp

    Arguments:
    -- moles_0: numpy array with all the initial moles organized by rows
    -- Temp: numpy array of the temperatures to be evaluated
    -- P: pressure of the system
    """

    allResults = pd.concat(generate_data_for_NN(mol_0, Temp, P) for mol_0 in moles_0)

    allResults.to_csv("data/data_for_nn.csv", index_label="Temperature")
