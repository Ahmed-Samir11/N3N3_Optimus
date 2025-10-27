# qubo_builder.py
import numpy as np
import pandas as pd

def build_qubo(df, budget=900000, max_grids=10, min_population=15000, 
               alpha=1e-1, gamma=1e-1, theta=1e-6, mu=2, lambda_=1e-2):
    """
    Build QUBO matrix for microgrid optimization.
    
    Args:
        df (pd.DataFrame): DataFrame with site data
        budget (float): Budget constraint
        max_grids (int): Maximum number of grids
        min_population (int): Minimum population coverage
        alpha (float): Population coverage weight
        gamma (float): Energy capacity weight
        theta (float): Budget penalty weight
        mu (float): Grid count penalty weight
        lambda_ (float): Population constraint penalty weight
    
    Returns:
        tuple: (QUBO_matrix, offset)
    """
    # Extract data
    install_costs = df["Installation_Cost_USD"].values
    population_coverage = df["Population_Coverage"].values
    energy_capacity = df["Energy_Capacity_kWh_day"].values
    num_sites = len(install_costs)
    
    # Initialize QUBO matrix
    Q = np.zeros((num_sites, num_sites))
    
    # Objective function terms
    for i in range(num_sites):
        # Linear terms: cost - alpha*population - gamma*energy
        Q[i, i] += install_costs[i] - alpha * population_coverage[i] - gamma * energy_capacity[i]
    
    # Budget constraint: theta * (sum(costs * x) - budget)^2
    for i in range(num_sites):
        Q[i, i] += theta * (install_costs[i]**2 - 2 * budget * install_costs[i])
        for j in range(num_sites):
            if i != j:
                Q[i, j] += 2 * theta * install_costs[i] * install_costs[j]
    
    # Grid count constraint: mu * (sum(x) - max_grids)^2
    for i in range(num_sites):
        Q[i, i] += mu * (1 - 2 * max_grids)
        for j in range(num_sites):
            if i != j:
                Q[i, j] += 2 * mu
    
    # Population constraint: lambda * (min_population - sum(population * x))^2
    for i in range(num_sites):
        Q[i, i] += lambda_ * (population_coverage[i]**2 - 2 * min_population * population_coverage[i])
        for j in range(num_sites):
            if i != j:
                Q[i, j] += 2 * lambda_ * population_coverage[i] * population_coverage[j]
    
    # Constant offset terms
    offset = theta * budget**2 + mu * max_grids**2 + lambda_ * min_population**2
    
    return Q, offset

def objective_function(x, df):
    """
    Calculate objective function value.
    
    Args:
        x (np.array): Binary solution vector
        df (pd.DataFrame): DataFrame with site data
    
    Returns:
        float: Objective function value
    """
    install_costs = df["Installation_Cost_USD"].values
    population_coverage = df["Population_Coverage"].values
    energy_capacity = df["Energy_Capacity_kWh_day"].values
    
    return np.sum(install_costs * x) - 1e-1 * np.sum(population_coverage * x) - 1e-1 * np.sum(energy_capacity * x)

def constraint_budget(x, df, budget=900000):
    """
    Calculate budget constraint violation.
    
    Args:
        x (np.array): Binary solution vector
        df (pd.DataFrame): DataFrame with site data
        budget (float): Budget constraint
    
    Returns:
        float: Budget constraint penalty
    """
    install_costs = df["Installation_Cost_USD"].values
    return 1e-6 * (np.sum(install_costs * x) - budget) ** 2

def constraint_grids(x, max_grids=10):
    """
    Calculate grid count constraint violation.
    
    Args:
        x (np.array): Binary solution vector
        max_grids (int): Maximum number of grids
    
    Returns:
        float: Grid count constraint penalty
    """
    return 2 * (np.sum(x) - max_grids) ** 2

def constraint_population(x, df, min_population=15000):
    """
    Calculate population constraint violation.
    
    Args:
        x (np.array): Binary solution vector
        df (pd.DataFrame): DataFrame with site data
        min_population (int): Minimum population coverage
    
    Returns:
        float: Population constraint penalty
    """
    population_coverage = df["Population_Coverage"].values
    return 1e-2 * (min_population - np.sum(population_coverage * x)) ** 2

def total_cost(x, df, budget=900000, max_grids=10, min_population=15000):
    """
    Calculate total cost including all constraints.
    
    Args:
        x (np.array): Binary solution vector
        df (pd.DataFrame): DataFrame with site data
        budget (float): Budget constraint
        max_grids (int): Maximum number of grids
        min_population (int): Minimum population coverage
    
    Returns:
        float: Total cost
    """
    obj = objective_function(x, df)
    budget_penalty = constraint_budget(x, df, budget)
    grids_penalty = constraint_grids(x, max_grids)
    population_penalty = constraint_population(x, df, min_population)
    
    return obj + budget_penalty + grids_penalty + population_penalty

def analyze_solution(x, df):
    """
    Analyze solution and return summary statistics.
    
    Args:
        x (np.array): Binary solution vector
        df (pd.DataFrame): DataFrame with site data
    
    Returns:
        dict: Solution analysis
    """
    selected_df = df[x == 1]
    
    if len(selected_df) == 0:
        return {
            "total_cost": 0,
            "total_population": 0,
            "total_energy": 0,
            "num_sites": 0,
            "selected_sites": []
        }
    
    total_cost = selected_df["Installation_Cost_USD"].sum()
    total_population = selected_df["Population_Coverage"].sum()
    total_energy = selected_df["Energy_Capacity_kWh_day"].sum()
    
    return {
        "total_cost": total_cost,
        "total_population": total_population,
        "total_energy": total_energy,
        "num_sites": len(selected_df),
        "selected_sites": selected_df.to_dict('records')
    } 