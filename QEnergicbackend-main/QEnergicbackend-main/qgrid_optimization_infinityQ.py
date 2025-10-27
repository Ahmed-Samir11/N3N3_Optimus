# %% [markdown]
# ## Mathematical formulation of the problem (QUBO Model Team3)
# 
# ### **Problem: Microgrid Placement and Load Balancing Optimization Problem**
# 
# (it’s a multi-objective optimisation using Quantum Annealing)
# 
# ## **Objective (Goals to optimise):**
# 
# - **Minimize installation cost**
# - **Maximize population/electricity coverage**
# - **Optimize microgrid placement**
# - **Maximize total energy served**
# 
# ## **Constraints** (to *enforce* via penalties):
# 
# - Budget limit: Total installation cost ≤ budget
# - Fixed number of microgrids to install (optional toggle)
# - Cover at least a minimum number of villages
#     - Weather (e.g., solar irradiance) can be integrated as a *weighting factor* or filter (OPTIONAL)
# 
# ## **Notation:**
# 
# Let:
# 
# - n: Number of candidate locations for microgrids
# - m: Number of villages
# - $x_i \in \{0, 1\}$:  1 if microgrid is installed at site i, 0 otherwise
# - $y_{ij} \in \{0, 1\}$: 1 if village j is assigned to microgrid i, 0 otherwise
# 
# Parameters:
# 
# - $C_i$: Installation cost at location $i$
# - $D_j$: Electricity demand or population of village $j$
# - $E_i$: Max energy output of grid $i$ (solar potential)
# - $dist_{ij}$: Distance between grid site  $i$ and village $j$
# - $R$: Maximum coverage radius
# - $B$: Total budget allowed
# - $K$: Maximum number of grids to install
# - $M$: Minimum number of villages to cover
# 
# ## **QUBO Objective Function:**
# 
# The QUBO function is of the form:
# 
# $\text{Minimize }$ $Q(x, y) = \underbrace{\sum_{i} C_i x_i}_{\text{(1) Cost}}$ - $\alpha \underbrace{\sum_{i,j} D_j y_{ij}}_{\text{(2) Total energy served}} 
# + \beta \underbrace{\sum_{i,j} y_{ij} \cdot dist_{ij}}_{\text{(3) Transmission penalty}}
# + \text{(4) Penalty terms for constraints}$
# 
# ## **Penalty Terms for Constraints:**
# 
# 1. **Village assigned only if grid is installed**
# 
# $P_1 = \gamma \sum_{i,j} y_{ij} (1 - x_i)$
# 
# 1. **Each village assigned to one grid only (coverage constraint)**
# 
# $P_2 = \delta \sum_j \left(1 - \sum_i y_{ij} \right)^2$
# 
# 1. **Energy capacity not exceeded**
#     
#     Let $D_j$: demand of village $j$
#     
#     Let $E_i$: capacity of grid $i$
#     
# 
# $P_3 = \eta \sum_i \left( \sum_j D_j y_{ij} - E_i x_i \right)^2$
# 
# 1. **Budget constraint**
# 
# $P_4 = \theta \left( \sum_i C_i x_i - B \right)^2$
# 
# 1. **Fixed number of grids (optional)**
# 
# $P_5 = \mu \left( \sum_i x_i - K \right)^2$
# 
# 1. **Minimum number of villages covered**
# 
# $P_6 = \lambda \left( M - \sum_j \sum_i y_{ij} \right)^2$
# 
# ### For the input Data:
# - Candidate grid locations
# - Village demands
# - Cost for the installation
# 
# ## **Full QUBO Expression**
# 
# Putting it all together:
# 
# $\begin{aligned}
# Q(x, y) =\ & \sum_i C_i x_i \quad\textcolor{gray}{\# install cost} \\
# & - \alpha \sum_{i,j} D_j y_{ij} \quad\textcolor{gray}{\# reward energy coverage} \\
# & + \beta \sum_{i,j} dist_{ij} y_{ij} \quad\textcolor{gray}{\# penalize far assignment} \\
# & + \gamma \sum_{i,j} y_{ij}(1 - x_i) \quad\textcolor{gray}{\# invalid assignment} \\
# & + \delta \sum_j \left(1 - \sum_i y_{ij} \right)^2 \quad\textcolor{gray}{\# single assignment} \\
# & + \eta \sum_i \left( \sum_j D_j y_{ij} - E_i x_i \right)^2 \quad\textcolor{gray}{\# overloading} \\
# & + \theta \left( \sum_i C_i x_i - B \right)^2 \quad\textcolor{gray}{\# budget} \\
# & + \mu \left( \sum_i x_i - K \right)^2 \quad\textcolor{gray}{\# grid limit (optional)} \\
# & + \lambda \left( M - \sum_j \sum_i y_{ij} \right)^2 \quad\textcolor{gray}{\# min coverage}
# \end{aligned}$
# 
# ## ✅ Reformulated Problem (Single Binary Variable)
# 
# ### **Given:**
# 
# - $n$: Candidate microgrid locations
# - $x_i \in \{0, 1\}=$ 1 if a microgrid is installed at location i, 0 otherwise
# - $C_i$: Installation cost at location i
# - $E_i$: Energy generation potential at location i (solar potential)
# - $P_i$: Estimated population coverage or demand that can be served from i
# - $dist_{ij}$: Distance from grid site i to village j, used to estimate $P_i$
# - $B$: Total budget
# - $K$: Max number of grids to install
# - $M$: Minimum total population to cover
# 
# ---
# 
# ### 🎯 **New Objective Function (All in x)**
# 
# We want to:
# 
# - **Minimize installation cost**
# - **Maximize covered population (or energy served)**
# - **Maximize total energy served**
# - $\min_x \left( \sum_i C_i x_i - \alpha \sum_i P_i x_i - \gamma \sum_i E_i x_i \right)$
# 
# Where:
# 
# - The term $\sum_i C_i x_i:$ total cost
# - The term $\sum_i P_i x_i$: estimated population/energy served from grid $i$
# - $\alpha:$ trade-off weight (controls how much you value coverage vs cost)
# 
# ---
# 
# ### 🔐 Constraints as Penalties
# 
# We now incorporate everything else via penalty terms:
# 
# 1. **Budget constraint**:
# 
# $P_{\text{budget}} = \theta \left( \sum_i C_i x_i - B \right)^2$
# 
# 1. **Fixed number of microgrids (optional)**:
# 
# $P_{\text{grid count}} = \mu \left( \sum_i x_i - K \right)^2$
# 
# 1. **Minimum population coverage**:
# 
# $P_{\text{coverage}} = \lambda \left( M - \sum_i P_i x_i \right)^2$
# 
# ---
# 
# ### 🧮 Full QUBO Expression (All in x)
# 
# $\begin{aligned}
# Q(x) =\ & \sum_i C_i x_i - \alpha \sum_i P_i x_i - \gamma \sum_i E_i x_i\\
# & + \theta \left( \sum_i C_i x_i - B \right)^2 \\
# & + \mu \left( \sum_i x_i - K \right)^2 \\
# & + \lambda \left( M - \sum_i P_i x_i \right)^2
# \end{aligned}$
# 
# This QUBO is fully quadratic in $x_i$, and ready to be turned into a matrix for quantum optimization.
# 
# ---
# 
# ## 🔎 Interpretation
# 
# This compact model:
# 
# - Keeps the **number of qubits manageable**
# - Avoids explicit assignment variables $y_{ij}$, reducing complexity
# - Infers village coverage from grid location quality $P_i$, which can be precomputed from your GIS/distance data

# %%
# import necessary libraries
import re  #To import python's regular expression module
import numpy as np
import time
from titanq import Model, Vtype, Target #(titanq=quantum inspired optimization framework)
import pandas as pd
start_time = time.time()

# %%
# DATA GENERATION STEP ------
np.random.seed(42) 

#Function to generate data
def generate_dataset(name, num_sites):
    site_ids = [f"Site_{i+1}" for i in range(num_sites)]
    install_costs = np.random.randint(15000, 50000, size=num_sites)
    population_coverage = np.random.randint(100, 1500, size=num_sites)
    solar_potential = np.round(np.random.uniform(3.5, 6.5, size=num_sites), 2) #amount of solar energy captured at each site
    energy_capacity = np.round(solar_potential * population_coverage * 0.3, 2) #Estimated energy capacity per site
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
df

# %%
# PROBLEM PARAMETERS ----
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

# %%
# QUBO DEFINITION -----
def build_qubo_matrix(install_costs, population_coverage, energy_capacity,
                      alpha, gamma, theta, mu, lambda_,
                      budget, max_grids, min_population):
    
    n = len(install_costs) # Number of sites
    Q = np.zeros((n, n))   #Initial Qubo matrix
    b = np.zeros(n)        #bias (for constraints)

    # Objective: linear terms
    for i in range(n):
        b[i] += install_costs[i] - alpha * population_coverage[i] - gamma * energy_capacity[i]

    # Constraint: budget (theta * (sum(ci*xi - B))^2)
    for i in range(n):
        for j in range(n):
            Q[i, j] += theta * install_costs[i] * install_costs[j]
        b[i] -= 2 * theta * budget * install_costs[i]

    # Constraint: max grids (mu * (sum(xi - G))^2)
    for i in range(n):
        for j in range(n):
            Q[i, j] += mu
        b[i] -= 2 * mu * max_grids

    # Constraint: min population (lambda * (Pmin - sum(pi*xi))^2)
    for i in range(n):
        for j in range(n):
            Q[i, j] += lambda_ * population_coverage[i] * population_coverage[j]
        b[i] -= 2 * lambda_ * min_population * population_coverage[i]

    return Q, b, n

Q, b, n = build_qubo_matrix(
    install_costs=install_costs,
    population_coverage=population_coverage,
    energy_capacity=energy_capacity,
    alpha=alpha,
    gamma=gamma,
    theta=theta,
    mu=mu,
    lambda_=lambda_,
    budget=budget,
    max_grids=max_grids,
    min_population=min_population
)


# %%
# SETTING UP THE TITANQ MODEL
TITANQ_DEV_API_KEY = "put_your_token"
model = Model(api_key=TITANQ_DEV_API_KEY)
x = model.add_variable_vector(name="x", size=n, vtype=Vtype.BINARY)
model.set_objective_matrices(weights=Q, bias=b, target=Target.MINIMIZE)


# %%
# Solver parameters
num_chains = 128     #number of parallel chains
num_engines = 4      # number of parallel processing units to use
coupling_mult = 0.4  ##inter-chain coupling multiplier, which affects how strongly 
                     #the different solution chains influence each other during optimization.

# temperature range 
T_min = 0.01
T_max = 1e3

end_time = time.time()
execution_time = end_time - start_time

beta = (1.0 / np.geomspace(T_min, T_max, num_chains)).tolist() #guides how “hot” or “cold” each chain is during optimization.

# Run optimization
results = model.optimize(
    beta=beta,
    coupling_mult=coupling_mult,
    timeout_in_secs=timeout_in_seconds,
    num_engines=num_engines,
    num_chains=num_chains
)

# %%
# PROCESS RESULTS
best_energy = float("inf")
best_vector = None
print("\nIsing Energy   | Solution")
print("-" * 40)

for energy, vec in results.result_items():
    print(f"{energy:.4f} | {vec}")
    if energy < best_energy:
        best_energy = energy
        best_vector = vec

# Extract best solution
selected_df = df[np.array(best_vector) == 1]

total_cost = selected_df["Installation_Cost_USD"].sum()
total_population = selected_df["Population_Coverage"].sum()
total_energy = selected_df["Energy_Capacity_kWh_day"].sum()
    
print("\nBest solution found:")
print("✅ Selected Microgrid Sites:")
print(selected_df[["Site_ID", "Installation_Cost_USD", "Population_Coverage", "Energy_Capacity_kWh_day", "X_coord", "Y_coord"]])
print("\n📊 Summary:")
print(f"   - Total Installation Cost: ${total_cost}")
print(f"   - Total Population Covered: {total_population} people")
print(f"   - Total Energy Capacity: {total_energy} kWh/day")
print(f"   - Execution Time: {execution_time:.2f} seconds")

# %%



