#!/usr/bin/env python3
"""Test script to check solver output."""

from robin_logistics import LogisticsEnvironment
from solver import my_solver
import json

env = LogisticsEnvironment()
result = my_solver(env)

print("=" * 80)
print("SOLVER OUTPUT SUMMARY")
print("=" * 80)
print(f"\nTotal routes: {len(result['routes'])}")

for idx, route in enumerate(result['routes'][:2], 1):  # Show first 2 routes
    print(f"\n--- Route {idx}: {route['vehicle_id']} ---")
    print(f"Total steps: {len(route['steps'])}")
    
    for step_idx, step in enumerate(route['steps'], 1):
        print(f"\n  Step {step_idx}: Node {step['node_id']}")
        print(f"    Pickups: {len(step['pickups'])}")
        print(f"    Deliveries: {len(step['deliveries'])}")
        print(f"    Unloads: {len(step['unloads'])}")
        
        if step['deliveries']:
            print(f"    Delivery details:")
            for delivery in step['deliveries']:
                print(f"      - Order {delivery['order_id']}: {delivery['sku_id']} x {delivery['quantity']}")

print("\n" + "=" * 80)
