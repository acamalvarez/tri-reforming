import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions import find_results, generate_data_for_NN, concat_results
from constants import Strings

plt.rcParams['axes.labelweight'] = 'bold'

#          CH4, CO2, H2O, O2,  CO,  H2,  C
moles_0 = [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
           [1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
           [1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0]]
Temp = np.arange(200, 1050, 50)
P = 1

# df = find_results(moles_0, Temp, P)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(df[['mol_CH4', 'mol_CO2', 'mol_H2O']], '-o')

# ax.set_xlabel(Strings.Temperature)
# ax.set_ylabel(Strings.mol)
# plt.legend([Strings.mol_CH4, Strings.mol_CO2, Strings.mol_H2O])
# plt.show()

# print(generate_data_for_NN(moles_0[0], Temp, P))

print(concat_results(moles_0, Temp, P))
