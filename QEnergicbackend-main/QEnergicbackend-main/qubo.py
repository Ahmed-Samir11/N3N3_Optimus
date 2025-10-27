import numpy as np
import pandas as pd
from autoqubo import SamplingCompiler, Utils
from autoqubo.symbolic import symbolic_matrix, insert_values
import time

# ---- STEP 1: DATA GENERATION ----
np.random.seed(42)

def generate_dataset(name, num_sites):
    site_ids = [f"Site_{i+1}" for i in range(num_sites)]
    install_costs = np.random.randint(15000, 50000, size=num_sites)
    population_coverage = np.random.randint(100, 1500, size=num_sites)
    solar_potential = np.round(np.random.uniform(3.5, 6.5, size=num_sites), 2)
    energy_capacity = np.round(solar_potential * population_coverage * 0.3, 2)
    coordinates = np.random.uniform(low=0.0, high=1.0, size=(num_sites, 2))
    df = pd.DataFrame({
        "Site_ID": site_ids,
        "Installation_Cost_USD": install_costs,
        "Population_Coverage": population_coverage,
        "Solar_Potential_kWh_m2_day": solar_potential,
        "Energy_Capacity_kWh_day": energy_capacity,
        "X_coord": coordinates[:, 0],
        "Y_coord": coordinates[:, 1]
    })
    return df

df = generate_dataset("Ethiopia_Offgrid_Potential", 50)

# ---- STEP 2: QUBO PARAMETERS ----
install_costs = df["Installation_Cost_USD"].values
population_coverage = df["Population_Coverage"].values
energy_capacity = df["Energy_Capacity_kWh_day"].values
num_sites = len(install_costs)

# Hyperparameters
alpha = 1e-1
gamma = 1e-1
theta = 1e-6
mu = 2
lambda_ = 1e-2

budget = 900000
max_grids = 10
min_population = 15000

# ---- STEP 3: QUBO DEFINITION ----
symbolic_vars = symbolic_matrix(1, num_sites, positive=True)

def objective(x):
    x = np.array(x).reshape(1, num_sites)[0]
    return np.sum(install_costs * x) - alpha * np.sum(population_coverage * x) - gamma * np.sum(energy_capacity * x)

def constraint_budget(x):
    x = np.array(x).reshape(1, num_sites)[0]
    return theta * (np.sum(install_costs * x) - budget) ** 2

def constraint_grids(x):
    x = np.array(x).reshape(1, num_sites)[0]
    return mu * (np.sum(x) - max_grids) ** 2

def constraint_population(x):
    x = np.array(x).reshape(1, num_sites)[0]
    return lambda_ * (min_population - np.sum(population_coverage * x)) ** 2

def total_cost(x):
    return objective(x) + constraint_budget(x) + constraint_grids(x) + constraint_population(x)

sym_qubo, offset = SamplingCompiler.generate_qubo(lambda x: total_cost(x), total_cost, 1 * num_sites)
pik = np.stack([install_costs, population_coverage, energy_capacity], axis=1).reshape(1, num_sites, 3)
explicit_qubo = insert_values(sym_qubo, pik)

# ---- STEP 4: SOLVE QUBO ----
start_time = time.time()
solutions, energies = Utils.solve(explicit_qubo, offset)
End_time = time.time()
execution_time = End_time - start_time

best_solution = solutions[np.argmin(energies)]
x_best = np.array(best_solution).reshape(1, num_sites)[0]

# ---- STEP 5: POST-SOLUTION ANALYSIS ----
selected_df = df[x_best == 1]
total_cost = selected_df["Installation_Cost_USD"].sum()
total_population = selected_df["Population_Coverage"].sum()
total_energy = selected_df["Energy_Capacity_kWh_day"].sum()

print("✅ Selected Microgrid Sites:")
print(selected_df[["Site_ID", "Installation_Cost_USD", "Population_Coverage", "Energy_Capacity_kWh_day", "X_coord", "Y_coord"]])
print("\n📊 Summary:")
print(f"   - Total Installation Cost: ${total_cost}")
print(f"   - Total Population Covered: {total_population} people")
print(f"   - Total Energy Capacity: {total_energy} kWh/day")
print(f"   - Execution Time: {execution_time:.2f} seconds")
