import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fuzzy Variables
solar_intensity = ctrl.Antecedent(np.arange(0, 1001, 1), 'solar_intensity')
energy_demand = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'energy_demand')
temperature = ctrl.Antecedent(np.arange(10, 61, 1), 'temperature')
solar_usage = ctrl.Consequent(np.arange(0, 101, 1), 'solar_usage')


