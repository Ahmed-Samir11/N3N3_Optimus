# data_generator.py
import numpy as np
import pandas as pd

def generate_dataset(name, num_sites, seed=42):
    """
    Generate dataset for microgrid optimization.
    
    Args:
        name (str): Dataset name
        num_sites (int): Number of sites to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: DataFrame with site data
    """
    np.random.seed(seed)
    
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

def generate_ethiopia_dataset(num_sites=50, seed=42):
    """
    Generate Ethiopia-specific dataset with real coordinates.
    
    Args:
        num_sites (int): Number of sites to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: DataFrame with Ethiopia site data
    """
    np.random.seed(seed)
    
    # Ethiopia bounding box coordinates
    ETHIOPIA_BBOX = [32.997583, 3.397448, 47.982379, 14.894053]
    min_lng, min_lat, max_lng, max_lat = ETHIOPIA_BBOX
    
    site_ids = [f"Site_{i+1}" for i in range(num_sites)]
    install_costs = np.random.randint(15000, 50000, size=num_sites)
    population_coverage = np.random.randint(100, 1500, size=num_sites)
    solar_potential = np.round(np.random.uniform(3.5, 6.5, size=num_sites), 2)
    energy_capacity = np.round(solar_potential * population_coverage * 0.3, 2)
    
    # Generate coordinates within Ethiopia bounding box
    lng_coords = np.random.uniform(min_lng, max_lng, size=num_sites)
    lat_coords = np.random.uniform(min_lat, max_lat, size=num_sites)
    
    df = pd.DataFrame({
        "Site_ID": site_ids,
        "Installation_Cost_USD": install_costs,
        "Population_Coverage": population_coverage,
        "Solar_Potential_kWh_m2_day": solar_potential,
        "Energy_Capacity_kWh_day": energy_capacity,
        "X_coord": lng_coords,  # longitude
        "Y_coord": lat_coords   # latitude
    })
    
    return df

if __name__ == "__main__":
    # Generate and display sample dataset
    df = generate_ethiopia_dataset(50)
    print("Generated Ethiopia Dataset:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Total installation cost: ${df['Installation_Cost_USD'].sum():,}")
    print(f"Total population coverage: {df['Population_Coverage'].sum():,}")
    print(f"Total energy capacity: {df['Energy_Capacity_kWh_day'].sum():.2f} kWh/day") 