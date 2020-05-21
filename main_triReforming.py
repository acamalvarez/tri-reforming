import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from DelG_f import Gibbs
from constants import Constants

#  CH4  CO2  H2O  O2    CO   H2   He   C
def found_moles(moles_0, T, P):

    '''
    calculates the moles at the equilibrium
    Arguments:
    -- moles_0: initial moles in this order: 
        CH4, CO2, H2O, O2, CO, H2, He, C
    -- T: temperature, °C
    -- P: pressure, atm
    Returns:
    -- numpy array with the moles at the equilibrium of 
    CH4, CO2, H2O, O2, CO, H2, He, C
    
    '''
    # calculates temperature and pressure to convinient units
    T = T + 273 # temperature, K
    P = 1  # pressure, bar

    # organizes the parameters (T and P) in a numpy array
    parameters = np.array([T, P])

    # calculates the vector of elemental balances
    beq = np.dot(Constants.Aeq, np.array(moles_0))

    print(beq)

    # Defines the equality constraints with the elemental balance
    eq_cons = {'type': 'eq', 'fun': lambda x: np.dot(Constants.Aeq, np.array(x)) - beq}

    # eq_cons = {'type': 'eq', 'fun': lambda x: np.array([np.sum(Constants.Aeq[0] * np.array(x)) - beq[0], 
    #                                                     np.sum(Constants.Aeq[1] * np.array(x)) - beq[1], 
    #                                                     np.sum(Constants.Aeq[2] * np.array(x)) - beq[2], 
    #                                                     np.sum(Constants.Aeq[3] * np.array(x)) - beq[3]])}

    options = {'ftol': 1e-1, 'disp':True, 'maxiter':1000}

    # solve the optimization problem
    result = minimize(Gibbs, Constants.n0, args=parameters, method='SLSQP',
                    bounds=Constants.bnds,
                      constraints=eq_cons, options=options)

    return result.x

#          CH4, CO2, H2O, O2,  CO,  H2,  He,  C
moles_0 = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.1, 0.0]
# Temp = np.arange(200, 1050, 50)
Temp = [550]
P = 1

moles = np.zeros((len(Temp), 8))
conversions = np.zeros((len(Temp), 8))

for i in np.arange(len(Temp)):

    moles[i] = found_moles(moles_0, Temp[i], P)

    conversions[i] = 100 * (moles_0 - moles[i]) / moles_0

df_moles = pd.DataFrame(data=moles, index=Temp, 
        columns=Constants.columns_moles)

df_conv = pd.DataFrame(data=conversions, index=Temp, 
        columns=Constants.columns_conversions)


# print(df_conv[['conv_CH4', 'conv_CO2', 'conv_H2O']])
print(df_moles)



def create_df(moles_0, Temperatures, P):
    pass


