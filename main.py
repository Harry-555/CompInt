import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qt5agg')

def solar_fuzzy_system_sim():
    # Fuzzy Variables
    solar_intensity = ctrl.Antecedent(np.arange(0, 1001, 1), 'solar_intensity')
    energy_demand = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'energy_demand')
    temperature = ctrl.Antecedent(np.arange(10, 61, 1), 'temperature')
    solar_usage = ctrl.Consequent(np.arange(0, 101, 1), 'solar_usage')

    # Input
    # Solar Intensity
    solar_intensity['low'] = fuzz.trapmf(solar_intensity.universe, [0, 0, 150, 400])
    solar_intensity['medium'] = fuzz.trimf(solar_intensity.universe, [300, 500, 700])
    solar_intensity['high'] = fuzz.trapmf(solar_intensity.universe, [600, 850, 1000, 1000])

    # Energy Demand
    energy_demand['low'] = fuzz.trimf(energy_demand.universe, [0, 0, 1.5])
    energy_demand['medium'] = fuzz.trimf(energy_demand.universe, [1, 2.5, 4])
    energy_demand['high'] = fuzz.trimf(energy_demand.universe, [3, 5, 5])

    # Temperature
    temperature['low'] = fuzz.trimf(temperature.universe, [10, 10, 22])
    temperature['medium'] = fuzz.trimf(temperature.universe, [18, 30, 42])
    temperature['high'] = fuzz.trimf(temperature.universe, [35, 60, 60])

    # Output (Solar Usage)
    solar_usage['low'] = fuzz.trapmf(solar_usage.universe, [0, 0, 20, 40])
    solar_usage['medium'] = fuzz.trimf(solar_usage.universe, [30, 50, 70])
    solar_usage['high'] = fuzz.trapmf(solar_usage.universe, [60, 80, 100, 100])

    # Set of Rules
    rules = [
        ctrl.Rule(solar_intensity['high'] & energy_demand['low'] & temperature['low'], solar_usage['high']),
        ctrl.Rule(solar_intensity['high'] & energy_demand['low'] & temperature['medium'], solar_usage['high']),
        ctrl.Rule(solar_intensity['high'] & energy_demand['medium'] & temperature['low'], solar_usage['high']),
        ctrl.Rule(solar_intensity['high'] & energy_demand['medium'] & temperature['medium'], solar_usage['high']),
        ctrl.Rule(solar_intensity['high'] & energy_demand['high'] & temperature['low'], solar_usage['high']),
        ctrl.Rule(solar_intensity['high'] & energy_demand['high'] & temperature['medium'], solar_usage['high']),
        ctrl.Rule(solar_intensity['medium'] & energy_demand['low'] & temperature['low'], solar_usage['high']),
        ctrl.Rule(solar_intensity['high'] & energy_demand['low'] & temperature['high'], solar_usage['medium']),
        ctrl.Rule(solar_intensity['high'] & energy_demand['medium'] & temperature['high'], solar_usage['medium']),
        ctrl.Rule(solar_intensity['high'] & energy_demand['high'] & temperature['high'], solar_usage['medium']),
        ctrl.Rule(solar_intensity['medium'] & energy_demand['medium'] & temperature['high'], solar_usage['low']),
        ctrl.Rule(solar_intensity['medium'] & energy_demand['high'] & temperature['high'], solar_usage['low']),
        ctrl.Rule(solar_intensity['medium'] & energy_demand['high'] & temperature['low'], solar_usage['medium']),
        ctrl.Rule(solar_intensity['medium'] & energy_demand['high'] & temperature['medium'], solar_usage['low']),
        ctrl.Rule(solar_intensity['low'] & energy_demand['low'] & temperature['high'], solar_usage['low']),
        ctrl.Rule(solar_intensity['low'] & energy_demand['medium'] & temperature['medium'], solar_usage['low']),
        ctrl.Rule(solar_intensity['low'] & energy_demand['medium'] & temperature['high'], solar_usage['low']),
        ctrl.Rule(solar_intensity['low'] & energy_demand['high'] & temperature['low'], solar_usage['low']),
        ctrl.Rule(solar_intensity['low'] & energy_demand['high'] & temperature['medium'], solar_usage['low']),
        ctrl.Rule(solar_intensity['low'] & energy_demand['high'] & temperature['high'], solar_usage['low']),
        ctrl.Rule(solar_intensity['medium'] & energy_demand['low'] & temperature['medium'], solar_usage['medium']),
        ctrl.Rule(solar_intensity['medium'] & energy_demand['low'] & temperature['high'], solar_usage['medium']),
        ctrl.Rule(solar_intensity['medium'] & energy_demand['medium'] & temperature['low'], solar_usage['medium']),
        ctrl.Rule(solar_intensity['medium'] & energy_demand['medium'] & temperature['medium'], solar_usage['medium']),
        ctrl.Rule(solar_intensity['low'] & energy_demand['low'] & temperature['low'], solar_usage['low']),
        ctrl.Rule(solar_intensity['low'] & energy_demand['low'] & temperature['medium'], solar_usage['low']),
        ctrl.Rule(solar_intensity['low'] & energy_demand['medium'] & temperature['low'], solar_usage['low']),
    ]

    # Construction & Simulation
    solar_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(solar_ctrl)

def solar_sim_test(sim, intensity, demand, temperature):
    sim.input['solar_intensity'] = intensity
    sim.input['energy_demand'] = demand
    sim.input['temperature'] = temperature

    sim.compute()
    print("Solar Usage Recommendation:", round(solar_sim.output['solar_usage'], 2), "%")

    solar_usage = next(c for c in sim.ctrl.consequents if c.label == 'solar_usage')
    solar_usage.view(sim=sim)
    plt.show()

# #3D Plot (Only for fixed energy demand = 3.0)
def plot3d(x, y, z):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
    ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='x', offset=x.max()*1.5, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='y', offset=y.max()*1.5, cmap='viridis', alpha=0.5)
    ax.view_init(30, 200)
    plt.show()

if __name__ == '__main__':
    solar_sim = solar_fuzzy_system_sim()

    # 2PM Test Case
    solar_sim_test(solar_sim, 870, 3.5, 41)

    # 3D Visualisation
    x_solar_temp, y_solar_temp = np.meshgrid(np.linspace(0, 1000, 20),
                                             np.linspace(10, 60, 20))
    z_solar_usage = np.zeros_like(x_solar_temp)

    for i in range(20):
        for j in range(20):
            solar_sim.input['solar_intensity'] = x_solar_temp[i, j]
            solar_sim.input['energy_demand'] = 3.0
            solar_sim.input['temperature'] = y_solar_temp[i, j]
            try:
                solar_sim.compute()
                z_solar_usage[i, j] = solar_sim.output['solar_usage']
            except:
                z_solar_usage[i, j] = 0

    plot3d(x_solar_temp, y_solar_temp, z_solar_usage)

