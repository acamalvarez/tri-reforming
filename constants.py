import pandas as pd
import numpy as np


class Constants:
    """Define constants for the problem"""

    R = 8.314e-3  # universal gas constant, kJ/mol.K
    P0 = 1 / 1.01325  # reference pressure, bar

    # Delta Gibs energy of formation,
    DelG_f_raw = pd.read_csv("DelG_f.csv", index_col="Compound")  # pandas dataframe
    DelG_f = DelG_f_raw.values  # numpy array

    # Matrix of elemental ba
    # lance
    Aeq = np.array(
        [[1, 1, 0, 0, 1, 0, 1], [4, 0, 2, 0, 0, 2, 0], [0, 2, 1, 2, 1, 0, 0]]  # carbon balance  # hydrogen balance
    )  # oxygen balance

    # Initial values for the optimization
    #### CH4, CO2, H2O, O2,  CO,  H2,  C
    n0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # Bounds for the optimization
    bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

    # moles columns naes of the results dataframe

    columns_mol = ["mol_CH4", "mol_CO2", "mol_H2O", "mol_O2", "mol_CO", "mol_H2", "mol_C"]

    # conversions columns naes of the results dataframe
    columns_conversions = ["conv_CH4", "conv_CO2", "conv_H2O", "conv_O2", "conv_CO", "conv_H2", "conv_C"]


class Strings:
    mol_CH4 = r"Mol CH$_4$"
    mol_CO2 = r"Mol CO$_2$"
    mol_H2O = r"Mol H$_2$O"

    mol = "Mol"
    Temperature = r"Temperature / $^\circ $C"
