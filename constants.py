import pandas as pd
import numpy as np

R = 8.314e-3  # universal gas constant, kJ/mol.K
P0 = 1 / 1.01325  # reference pressure, bar
# Delta Gibs energy of formation,
del_G_f = pd.read_csv("data/DelG_f.csv", index_col="Compound").values  # numpy array dataframe

# Matrix of elemental balance
Aeq = np.array(
    [
        [1, 1, 0, 0, 1, 0, 1], # carbon balance
        [4, 0, 2, 0, 0, 2, 0], # hydrogen balance
        [0, 2, 1, 2, 1, 0, 0], # oxygen balance
    ]
)

# Initial values for the optimization
#### CH4, CO2, H2O, O2,  CO,  H2,  C
n0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Bounds for the optimization
BOUNDS = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

# moles columns naes of the results dataframe
COLUMNS_MOL = ["mol_CH4", "mol_CO2", "mol_H2O", "mol_O2", "mol_CO", "mol_H2", "mol_C"]
    
# conversions columns naes of the results dataframe
COLUMNS_CONVERSIONS = ["conv_CH4", "conv_CO2", "conv_H2O", "conv_O2", "conv_CO", "conv_H2", "conv_C"]


MOL_STRING = "Mol"
MOL_CH4_STRING = r"Mol CH$_4$"
MOL_CO2_STRING = r"Mol CO$_2$"
MOL_H2O_STRING = r"Mol H$_2$O"
TEMPERATURE_STRING = r"Temperature / $^\circ $C"
