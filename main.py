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

# 3 Input
    # Solar Intensity
solar_intensity['low'] = fuzz.trapmf(solar_intensity.universe, [0, 0, 150, 400])
solar_intensity['medium'] = fuzz.trimf(solar_intensity.universe, [300, 500, 700])
solar_intensity['high'] = fuzz.trapmf(solar_intensity.universe, [600, 850, 1000, 1000])
