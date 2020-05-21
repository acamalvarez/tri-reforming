import pandas as pd
import numpy as np

class Constants:
    '''Define constants for the problem
    '''
    R = 8.314 # universal gas constant, Pa.m^3/mol.K
    P0 = 1  # reference pressure, bar

    # Delta Gibs energy of formation, 
    DelG_f_raw = pd.read_csv('DelG_f.csv', index_col='Compound') # pandas dataframe
    DelG_f = DelG_f_raw.values # numpy array

    # Matrix of elemental balance
    Aeq = np.array([[1, 1, 0, 0, 1, 0, 0, 1], # carbon balance
                [4, 0, 2, 0, 0, 2, 0, 0], # hydrogen balance
                [0, 2, 1, 2, 1, 0, 0, 0], # oxygen balance
                [0, 0, 0, 0, 0, 0, 1, 0]]) # helium balance

    #          CH4, CO2, H2O, O2,  CO,  H2,  He,  C
    moles_0 = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Initial values for the optimization
    n0 = [0.3, 0.1, 0.5, 0.01, 0.5, 0.5, 0.01, 0.01]

    # Bounds for the optimization
    bnds = ((0, None), (0, None), (0, None), (0, None), 
            (0, None), (0, None), (0, None), (0, None))

    columns_moles = ['mol_CH4', 'mol_CO2', 'mol_H2O', 'mol_O2', 
                    'mol_CO', 'mol_H2', 'mol_He', 'mol_C']

    columns_conversions = ['conv_CH4', 'conv_CO2', 'conv_H2O', 'conv_O2', 
                            'conv_CO', 'conv_H2', 'conv_He', 'conv_C']



class Critical_constants:

    # critical temperature, K
    Tc = np.array([[190.55, 303.95, 647.15, 154.6, 132.8, 33.3, 5.19]])
    # critical pressure, Pa
    Pc = np.array([[4.64, 7.39, 22.1, 5.08, 3.5, 1.3, 0.22]]) * 1e6
    # acentric factor
    w = np.array([[0.008, 0.225, 0.344, 0.021, 0.049, -0.22, -0.387]])
