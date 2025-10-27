#!/usr/bin/env python3

#pip install robin-logistics-env  --- Before first run install in the terminal
"""
Contestant solver for the Robin Logistics Environment.
 
Generates a valid solution using basic assignment and BFS-based routing.
"""
from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional
 
 
def solver(env) -> Dict:
    """Generate a simple, valid solution using the road network.
 
    Args:
        env: LogisticsEnvironment instance
 
    Returns:
        A complete solution dict with routes and sequential steps.
    """
    solution = {"routes": []}
 
    order_ids: List[str] = env.get_all_order_ids()
    available_vehicle_ids: List[str] = env.get_available_vehicles()
    #road_network = env.get_road_network_data()
   # adjacency_list = road_network.get("adjacency_list", {})
    warehouses = env.warehouses
    print(warehouses)
    
    return solution
 
if __name__ == '__main__':
    env= LogisticsEnviroment()
    my_solver(env)
 
