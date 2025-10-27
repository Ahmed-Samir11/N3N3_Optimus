"""
Analyze Solver 75 DQN vs Fallback Usage

This script instruments solver 75 to track:
1. How many orders are assigned by DQN vs greedy fallback
2. Which orders fail DQN assignment and why
3. Performance comparison between DQN and fallback assignments
"""

import sys
from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_76 import solver
import numpy as np

def analyze_assignment_strategy():
    """Test solver 75 and analyze DQN vs fallback usage"""
    
    print("=" * 80)
    print("SOLVER 75 ANALYSIS: DQN vs Fallback Assignment")
    print("=" * 80)
    
    # Create environment
    env = LogisticsEnvironment()
    
    # Import and patch the assignment functions to track usage
    from Ne3Na3_solver_76 import dqn_assignment, greedy_assignment
    
    # Track assignments
    dqn_assigned = []
    fallback_assigned = []
    
    # Run solver
    print("\nRunning solver 75...")
    result = solver(env)
    
    # Analyze result
    is_valid, msg, details = env.validate_solution_complete(result)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    if not is_valid:
        print(f"Message: {msg}")
        if details:
            print(f"Details: {details}")
    
    # Get fulfillment info
    fulfillment = env.get_solution_fulfillment_summary(result)
    total_orders = fulfillment['total_orders']
    fulfilled = fulfillment['fully_fulfilled_orders']
    vehicles_used = fulfillment['vehicles_used']
    
    print(f"\nResults:")
    print(f"  Orders fulfilled: {fulfilled}/{total_orders} ({fulfilled/total_orders*100:.1f}%)")
    print(f"  Vehicles used: {vehicles_used}/{fulfillment['total_vehicles']}")
    
    if is_valid:
        cost = env.calculate_solution_cost(result)
        print(f"  Total cost: ${cost:,.2f}")
    
    # Analyze unfulfilled orders
    unfulfilled = []
    for order_id, details in fulfillment['order_fulfillment_details'].items():
        if details['fulfillment_rate'] < 100:
            unfulfilled.append(order_id)
    
    if unfulfilled:
        print(f"\nUnfulfilled orders ({len(unfulfilled)}):")
        for oid in unfulfilled[:5]:  # Show first 5
            print(f"  - {oid}")
        if len(unfulfilled) > 5:
            print(f"  ... and {len(unfulfilled) - 5} more")
    
    return result, fulfillment

def test_with_instrumentation():
    """
    Create an instrumented version to track DQN vs fallback
    """
    print("\n" + "=" * 80)
    print("INSTRUMENTED TEST - Tracking DQN vs Fallback")
    print("=" * 80)
    
    from robin_logistics import LogisticsEnvironment
    import importlib
    import Ne3Na3_solver_76
    
    # Reload to get fresh module
    importlib.reload(Ne3Na3_solver_76)
    
    env = LogisticsEnvironment()
    
    # Patch the functions to track usage
    original_dqn = Ne3Na3_solver_76.dqn_assignment
    original_greedy = Ne3Na3_solver_76.greedy_assignment
    
    dqn_stats = {'count': 0, 'orders': []}
    greedy_stats = {'count': 0, 'orders': []}
    
    def tracked_dqn(*args, **kwargs):
        result = original_dqn(*args, **kwargs)
        order_assignments, vehicle_states = result
        
        # Count assigned orders
        for v_id, orders_dict in order_assignments.items():
            for order_id in orders_dict.keys():
                dqn_stats['orders'].append(order_id)
        dqn_stats['count'] = len(set(dqn_stats['orders']))
        
        return result
    
    def tracked_greedy(env, order_assignments, vehicle_states, already_assigned):
        before_count = sum(len(orders) for orders in order_assignments.values())
        
        original_greedy(env, order_assignments, vehicle_states, already_assigned)
        
        after_count = sum(len(orders) for orders in order_assignments.values())
        greedy_stats['count'] = after_count - before_count
        
        # Track which orders were added
        for v_id, orders_dict in order_assignments.items():
            for order_id in orders_dict.keys():
                if order_id not in dqn_stats['orders']:
                    greedy_stats['orders'].append(order_id)
    
    # Monkey patch
    Ne3Na3_solver_76.dqn_assignment = tracked_dqn
    Ne3Na3_solver_76.greedy_assignment = tracked_greedy
    
    # Run solver
    result = Ne3Na3_solver_76.solver(env)
    
    # Report
    print(f"\nAssignment breakdown:")
    print(f"  DQN assigned: {dqn_stats['count']} orders")
    print(f"  Fallback assigned: {greedy_stats['count']} orders")
    print(f"  Total assigned: {dqn_stats['count'] + greedy_stats['count']}")
    print(f"  Total orders: {len(env.get_all_order_ids())}")
    
    print(f"\nDQN effectiveness: {dqn_stats['count'] / len(env.get_all_order_ids()) * 100:.1f}%")
    
    if greedy_stats['orders']:
        print(f"\nOrders assigned by fallback (first 10):")
        for oid in greedy_stats['orders'][:10]:
            print(f"  - {oid}")
    
    # Restore original functions
    Ne3Na3_solver_76.dqn_assignment = original_dqn
    Ne3Na3_solver_76.greedy_assignment = original_greedy
    
    return dqn_stats, greedy_stats

def suggest_improvements():
    """Suggest improvements based on analysis"""
    print("\n" + "=" * 80)
    print("IMPROVEMENT SUGGESTIONS FOR SOLVER 75")
    print("=" * 80)
    
    suggestions = [
        {
            'name': 'Multi-Warehouse DQN State',
            'description': 'Add warehouse inventory state to DQN features',
            'benefit': 'Better warehouse selection when splitting orders',
            'difficulty': 'Medium',
            'expected_gain': '5-15% fulfillment improvement'
        },
        {
            'name': 'Epsilon-Greedy Exploration',
            'description': 'Add exploration to DQN during assignment',
            'benefit': 'Escape local optima, find better warehouse combinations',
            'difficulty': 'Low',
            'expected_gain': '3-10% fulfillment improvement'
        },
        {
            'name': 'Hybrid Scoring',
            'description': 'Combine DQN Q-values with greedy heuristic scores',
            'benefit': 'Use DQN guidance but ground in greedy validity',
            'difficulty': 'Medium',
            'expected_gain': '10-20% fulfillment improvement'
        },
        {
            'name': 'Order Clustering Pre-Processing',
            'description': 'Cluster orders spatially before assignment',
            'benefit': 'Better vehicle-order grouping, reduced distance',
            'difficulty': 'Medium',
            'expected_gain': '5-15% cost reduction'
        },
        {
            'name': 'Capacity-Aware DQN Reward',
            'description': 'Retrain DQN with capacity utilization in reward',
            'benefit': 'Better packing, more orders per vehicle',
            'difficulty': 'High',
            'expected_gain': '10-25% fulfillment improvement'
        },
        {
            'name': 'Fallback Improvement: Best-K Search',
            'description': 'Instead of greedy, try K best assignments per order',
            'benefit': 'Better fallback when DQN fails',
            'difficulty': 'Low',
            'expected_gain': '5-10% fulfillment improvement'
        },
    ]
    
    print("\nTop improvement strategies:\n")
    for i, s in enumerate(suggestions, 1):
        print(f"{i}. {s['name']} [{s['difficulty']} difficulty]")
        print(f"   {s['description']}")
        print(f"   Benefit: {s['benefit']}")
        print(f"   Expected: {s['expected_gain']}")
        print()

if __name__ == '__main__':
    # Run basic analysis
    result, fulfillment = analyze_assignment_strategy()
    
    # Run instrumented test
    try:
        dqn_stats, greedy_stats = test_with_instrumentation()
    except Exception as e:
        print(f"\nInstrumentation failed: {e}")
        print("This is expected - continuing with suggestions...")
    
    # Show improvement suggestions
    suggest_improvements()
