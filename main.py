import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions import find_results, generate_data_for_NN, csv_for_NN
from constants import Strings

plt.rcParams["axes.labelweight"] = "bold"

# moles_0 =  np.genfromtxt('moles_0.csv', delimiter=',', skip_header=1)

moles_0 = [0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]

Temp = range(500, 1050, 50)
P = 1

results = find_results(moles_0, Temp, P)

print(results)
