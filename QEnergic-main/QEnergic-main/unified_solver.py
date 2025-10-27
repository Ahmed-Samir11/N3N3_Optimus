#!/usr/bin/env python3
# unified_solver.py
"""
Unified solver that demonstrates the refactored codebase using the new data structure.
This script shows how to use the new data generator, QUBO builder, and solvers.
"""

import numpy as np
import pandas as pd
import json
import time
import argparse
import subprocess
import tempfile
import os
from data_generator import generate_ethiopia_dataset
from qubo_builder import build_qubo, analyze_solution

def nar_greedy_solver(df, budget, max_grids=10, record_progress=False):
    """
    NAR (Nearest Available Resource) greedy solver.
    Selects sites based on population-to-cost ratio within budget.
    If record_progress is True, saves progress to 'nar_progress.json'.
    """
    import json
    # Calculate population-to-cost ratio
    df_copy = df.copy()
    df_copy['pop_cost_ratio'] = df_copy['Population_Coverage'] / df_copy['Installation_Cost_USD']
    
    # Sort by ratio (descending)
    df_sorted = df_copy.sort_values('pop_cost_ratio', ascending=False)
    
    selected_indices = []
    total_cost = 0
    total_population = 0
    progress = []
    
    for idx, row in df_sorted.iterrows():
        if (total_cost + row['Installation_Cost_USD'] <= budget and 
            len(selected_indices) < max_grids):
            selected_indices.append(idx)
            total_cost += row['Installation_Cost_USD']
            total_population += row['Population_Coverage']
            # Record progress
            if record_progress:
                solution = [1 if i in selected_indices else 0 for i in range(len(df))]
                progress.append({
                    'step': len(selected_indices),
                    'solution': solution,
                    'total_cost': total_cost,
                    'total_population': total_population
                })
    # Save progress if requested
    if record_progress:
        with open('nar_progress.json', 'w') as f:
            json.dump(progress, f)
    # Create binary solution vector
    solution = np.zeros(len(df))
    for idx in selected_indices:
        solution[idx] = 1
    
    return solution, {
        'total_cost': total_cost,
        'total_population': total_population,
        'num_sites': len(selected_indices)
    }

def gurobi_solver(df, budget, max_grids=10, min_population=15000):
    """
    Gurobi solver using the new data structure.
    """
    # Build QUBO
    Q, offset = build_qubo(df, budget, max_grids, min_population)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as qubo_file:
        json.dump({'Q': Q.tolist()}, qubo_file)
        qubo_path = qubo_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as costs_file:
        json.dump(df['Installation_Cost_USD'].tolist(), costs_file)
        costs_path = costs_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
        json.dump(df.to_dict('records'), data_file)
        data_path = data_file.name
    
    try:
        # Run Gurobi solver
        result = subprocess.run([
            'python3', 'gurobi_optimize.py',
            '--qubo_file', qubo_path,
            '--budget', str(budget),
            '--costs_file', costs_path,
            '--data_file', data_path
        ], capture_output=True, text=True, check=True)
        
        # Parse result
        output_lines = result.stdout.strip().split('\n')
        result_data = json.loads(output_lines[-1])
        
        # Create binary solution vector
        solution = np.zeros(len(df))
        for idx in result_data['selected_indices']:
            solution[idx] = 1
        
        return solution, result_data
        
    except subprocess.CalledProcessError as e:
        print(f"Gurobi solver error: {e}")
        print(f"stderr: {e.stderr}")
        return None, None
    finally:
        # Clean up temporary files
        for path in [qubo_path, costs_path, data_path]:
            try:
                os.unlink(path)
            except:
                pass

def sa_solver(df, budget, max_grids=10, min_population=15000, steps=10000, tmax=25000.0, tmin=0.001, record_progress=False, progress_interval=1):
    """
    Simulated Annealing solver using the new data structure.
    """
    # Build QUBO
    Q, offset = build_qubo(df, budget, max_grids, min_population)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as qubo_file:
        json.dump({'Q': Q.tolist()}, qubo_file)
        qubo_path = qubo_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as costs_file:
        json.dump(df['Installation_Cost_USD'].tolist(), costs_file)
        costs_path = costs_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
        json.dump(df.to_dict('records'), data_file)
        data_path = data_file.name
    
    try:
        # Run SA solver
        cmd = [
            'python3', 'sa_optimize.py',
            '--qubo_file', qubo_path,
            '--budget', str(budget),
            '--costs_file', costs_path,
            '--data_file', data_path,
            '--steps', str(steps),
            '--tmax', str(tmax),
            '--tmin', str(tmin)
        ]
        if record_progress:
            cmd.append('--record_progress')
            cmd.extend(['--progress_interval', str(progress_interval)])
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse result
        output_lines = result.stdout.strip().split('\n')
        result_data = json.loads(output_lines[-1])
        
        # Create binary solution vector
        solution = np.zeros(len(df))
        for idx in result_data['selected_indices']:
            solution[idx] = 1
        
        return solution, result_data
        
    except subprocess.CalledProcessError as e:
        print(f"SA solver error: {e}")
        print(f"stderr: {e.stderr}")
        return None, None
    finally:
        # Clean up temporary files
        for path in [qubo_path, costs_path, data_path]:
            try:
                os.unlink(path)
            except:
                pass

def tabu_search_solver(df, budget, max_grids=10, min_population=15000, iterations=1000, tenure=10):
    """
    Tabu Search solver using the new data structure.
    """
    # Build QUBO
    Q, offset = build_qubo(df, budget, max_grids, min_population)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as qubo_file:
        json.dump({'Q': Q.tolist()}, qubo_file)
        qubo_path = qubo_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
        json.dump(df.to_dict('records'), data_file)
        data_path = data_file.name
    
    try:
        # Run Tabu Search solver
        result = subprocess.run([
            'python3', 'tabu_search_optimize.py',
            '--qubo_file', qubo_path,
            '--data_file', data_path,
            '--iterations', str(iterations),
            '--tenure', str(tenure)
        ], capture_output=True, text=True, check=True)
        
        # Parse result
        output_lines = result.stdout.strip().split('\n')
        result_data = json.loads(output_lines[-1])
        
        # Create binary solution vector
        solution = np.zeros(len(df))
        for idx in result_data['selected_indices']:
            solution[idx] = 1
        
        return solution, result_data
        
    except subprocess.CalledProcessError as e:
        print(f"Tabu Search solver error: {e}")
        print(f"stderr: {e.stderr}")
        return None, None
    finally:
        # Clean up temporary files
        for path in [qubo_path, data_path]:
            try:
                os.unlink(path)
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description='Unified microgrid optimization solver')
    parser.add_argument('--solver', choices=['nar', 'gurobi', 'sa', 'tabu', 'all'], 
                       default='all', help='Solver to use')
    parser.add_argument('--budget', type=float, default=900000, 
                       help='Budget constraint')
    parser.add_argument('--max_grids', type=int, default=10, 
                       help='Maximum number of grids')
    parser.add_argument('--min_population', type=int, default=15000, 
                       help='Minimum population coverage')
    parser.add_argument('--num_sites', type=int, default=50, 
                       help='Number of sites to generate')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--sa_steps', type=int, default=10000, 
                       help='Number of SA steps')
    parser.add_argument('--sa_tmax', type=float, default=25000.0, 
                       help='SA initial temperature')
    parser.add_argument('--sa_tmin', type=float, default=0.001, 
                       help='SA final temperature')
    parser.add_argument('--record_progress', action='store_true', help='Record progress for visualization')
    parser.add_argument('--progress_interval', type=int, default=1, help='Record progress every N steps (SA only)')
    parser.add_argument('--tabu_iterations', type=int, default=1000,
                       help='Tabu Search iterations')
    parser.add_argument('--tabu_tenure', type=int, default=10,
                       help='Tabu Search tenure')
    
    args = parser.parse_args()
    
    # Generate dataset
    print("Generating Ethiopia dataset...")
    df = generate_ethiopia_dataset(args.num_sites, args.seed)
    print(f"Generated {len(df)} sites")
    print(f"Total potential cost: ${df['Installation_Cost_USD'].sum():,}")
    print(f"Total potential population: {df['Population_Coverage'].sum():,}")
    print(f"Total potential energy: {df['Energy_Capacity_kWh_day'].sum():.2f} kWh/day")
    print()
    
    solvers = []
    if args.solver == 'all':
        solvers = ['nar', 'gurobi', 'sa', 'tabu']
    else:
        solvers = [args.solver]
    
    results = {}
    
    for solver_name in solvers:
        print(f"Running {solver_name.upper()} solver...")
        start_time = time.time()
        
        if solver_name == 'nar':
            solution, result = nar_greedy_solver(df, args.budget, args.max_grids, record_progress=True)
        elif solver_name == 'gurobi':
            solution, result = gurobi_solver(df, args.budget, args.max_grids, args.min_population)
        elif solver_name == 'sa':
            solution, result = sa_solver(df, args.budget, args.max_grids, args.min_population, 
                                       args.sa_steps, args.sa_tmax, args.sa_tmin, record_progress=args.record_progress, progress_interval=args.progress_interval)
        elif solver_name == 'tabu':
            solution, result = tabu_search_solver(df, args.budget, args.max_grids, args.min_population,
                                                  args.tabu_iterations, args.tabu_tenure)
        
        elapsed_time = time.time() - start_time
        
        if solution is not None:
            # Analyze solution
            analysis = analyze_solution(solution, df)
            results[solver_name] = {
                'solution': solution,
                'result': result,
                'analysis': analysis,
                'time': elapsed_time
            }
            
            print(f"✅ {solver_name.upper()} completed in {elapsed_time:.2f} seconds")
            print(f"   Selected sites: {analysis['num_sites']}")
            print(f"   Total cost: ${analysis['total_cost']:,}")
            print(f"   Total population: {analysis['total_population']:,}")
            print(f"   Total energy: {analysis['total_energy']:.2f} kWh/day")
        else:
            print(f"❌ {solver_name.upper()} failed")
        
        print()
    
    # Summary
    if len(results) > 1:
        print("📊 Summary:")
        print("-" * 80)
        print(f"{'Solver':<12} {'Sites':<6} {'Cost':<12} {'Population':<12} {'Energy':<12} {'Time':<8}")
        print("-" * 80)
        for solver_name, result in results.items():
            analysis = result['analysis']
            print(f"{solver_name.upper():<12} {analysis['num_sites']:<6} "
                  f"${analysis['total_cost']:<11,} {analysis['total_population']:<12,} "
                  f"{analysis['total_energy']:<11.2f} {result['time']:<8.2f}s")
    
    return results

if __name__ == '__main__':
    main() 