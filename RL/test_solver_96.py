"""
Test solver 96 (ULTRA-AGGRESSIVE consolidation) against solver 84
"""

from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_84 import solver as solver84
from Ne3Na3_solver_96 import solver as solver96
import time
import random
import numpy as np

def test_solver(solver_func, solver_name, seed):
    """Test a single solver"""
    random.seed(seed)
    np.random.seed(seed)
    env = LogisticsEnvironment()
    
    start = time.time()
    result = solver_func(env)
    elapsed = time.time() - start
    
    validation_result = env.validate_solution_complete(result)
    is_valid = validation_result[0] if isinstance(validation_result, tuple) else validation_result
    msg = validation_result[1] if isinstance(validation_result, tuple) and len(validation_result) > 1 else ""
    
    if not is_valid:
        print(f"  {solver_name} INVALID: {msg}")
        return None
    
    success, exec_msg = env.execute_solution(result)
    fulfillment = env.get_solution_fulfillment_summary(result)
    cost = env.calculate_solution_cost(result)
    
    fulfilled = fulfillment.get("fully_fulfilled_orders", 0)
    total_orders = len(env.get_all_order_ids())
    
    # Count vehicles used
    vehicles_used = sum(1 for v_route in result["routes"] if len(v_route["steps"]) > 2)
    
    return {
        "cost": cost,
        "vehicles": vehicles_used,
        "fulfilled": fulfilled,
        "total": total_orders,
        "time": elapsed,
        "success": success
    }

def main():
    seeds = [42, 100, 200, 300, 400, 500]
    
    print("="*80)
    print("ULTRA-AGGRESSIVE CONSOLIDATION TEST: Solver 84 vs Solver 96")
    print("="*80)
    
    solver84_results = []
    solver96_results = []
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"SEED: {seed}")
        print(f"{'='*80}\n")
        
        # Test solver 84
        print("[Solver 84 - Baseline DQN]")
        result84 = test_solver(solver84, "Solver 84", seed)
        if result84:
            print(f"  Fulfillment: {result84['fulfilled']}/{result84['total']}")
            print(f"  Cost: ${result84['cost']:,.2f}")
            print(f"  Vehicles: {result84['vehicles']}")
            print(f"  Time: {result84['time']:.2f}s")
            solver84_results.append(result84)
        
        # Test solver 96
        print("\n[Solver 96 - ULTRA-AGGRESSIVE Consolidation]")
        result96 = test_solver(solver96, "Solver 96", seed)
        if result96:
            print(f"  Fulfillment: {result96['fulfilled']}/{result96['total']}")
            print(f"  Cost: ${result96['cost']:,.2f}")
            print(f"  Vehicles: {result96['vehicles']}")
            print(f"  Time: {result96['time']:.2f}s")
            solver96_results.append(result96)
        
        # Compare
        if result84 and result96:
            print("\n[Impact]")
            cost_diff = result84["cost"] - result96["cost"]
            cost_pct = (cost_diff / result84["cost"]) * 100
            vehicle_diff = result84["vehicles"] - result96["vehicles"]
            fulfill_maintained = result96["fulfilled"] == result96["total"]
            
            print(f"  Cost savings: ${cost_diff:+,.2f} ({cost_pct:+.1f}%)")
            print(f"  Vehicle reduction: {vehicle_diff:+d}")
            print(f"  Fulfillment maintained: {fulfill_maintained}")
            
            if cost_pct >= 3.0:
                print(f"  ✅ SUCCESS: {cost_pct:.1f}% cost reduction!")
            elif cost_pct >= 1.0:
                print(f"  💰 GOOD: {cost_pct:.1f}% cost reduction")
            elif cost_pct > 0:
                print(f"  ⚠️ MARGINAL: {cost_pct:.1f}% cost reduction")
            else:
                print(f"  ❌ Cost increased by {abs(cost_pct):.1f}%")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    avg_cost_84 = sum(r["cost"] for r in solver84_results) / len(solver84_results)
    avg_cost_96 = sum(r["cost"] for r in solver96_results) / len(solver96_results)
    avg_vehicles_84 = sum(r["vehicles"] for r in solver84_results) / len(solver84_results)
    avg_vehicles_96 = sum(r["vehicles"] for r in solver96_results) / len(solver96_results)
    
    print(f"Solver 84 Average: ${avg_cost_84:,.2f}, {avg_vehicles_84:.1f} vehicles")
    print(f"Solver 96 Average: ${avg_cost_96:,.2f}, {avg_vehicles_96:.1f} vehicles")
    print()
    
    savings = avg_cost_84 - avg_cost_96
    savings_pct = (savings / avg_cost_84) * 100
    vehicle_reduction = avg_vehicles_84 - avg_vehicles_96
    
    print(f"Cost Savings: ${savings:+,.2f} ({savings_pct:+.1f}%)")
    print(f"Vehicle Reduction: {vehicle_reduction:+.2f}")
    
    # Win rate
    wins = sum(1 for r84, r96 in zip(solver84_results, solver96_results) 
               if r84["cost"] > r96["cost"])
    print(f"\nWin Rate: {wins}/{len(solver84_results)} scenarios")
    
    # Check fulfillment
    all_fulfilled = all(r["fulfilled"] == r["total"] for r in solver96_results)
    print(f"Fulfillment: {'✅ Maintained' if all_fulfilled else '⚠️ Some orders unfulfilled'}")
    
    if savings_pct >= 3.0:
        print(f"\n✅ SUCCESS: {savings_pct:.1f}% cost reduction")
    elif savings_pct >= 1.0:
        print(f"\n💰 GOOD: {savings_pct:.1f}% cost reduction")
    elif savings_pct > 0:
        print(f"\n⚠️ MARGINAL: {savings_pct:.1f}% improvement")
    else:
        print(f"\n❌ NO IMPROVEMENT: {savings_pct:.1f}% change")

if __name__ == "__main__":
    main()
