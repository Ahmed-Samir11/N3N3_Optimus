#!/usr/bin/env python3
"""
Run headless solver to validate the solution.
"""

from robin_logistics import LogisticsEnvironment
from solver import my_solver

def main():
    print("=" * 80)
    print("RUNNING SOLVER IN HEADLESS MODE")
    print("=" * 80)
    
    env = LogisticsEnvironment()
    env.set_solver(my_solver)
    
    print("\nGenerating solution...")
    result = env.run_headless("test_run_001")
    
    print(f"\nResult: {result}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
