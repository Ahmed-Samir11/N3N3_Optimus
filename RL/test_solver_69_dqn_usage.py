"""
Test solver 69 to see when DQN is used vs fallback
"""

from robin_logistics import LogisticsEnvironment
import importlib
import Ne3Na3_solver_69

# Reload to get fresh module
importlib.reload(Ne3Na3_solver_69)

env = LogisticsEnvironment()

# Patch the functions to track usage
original_dqn = Ne3Na3_solver_69.dqn_assignment
original_greedy = Ne3Na3_solver_69.greedy_assignment

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
Ne3Na3_solver_69.dqn_assignment = tracked_dqn
Ne3Na3_solver_69.greedy_assignment = tracked_greedy

# Run solver
print("Running solver 69 with instrumentation...\n")
result = Ne3Na3_solver_69.solver(env)

# Report
print("=" * 70)
print("SOLVER 69 - DQN vs FALLBACK USAGE")
print("=" * 70)
print(f"\nDQN assigned:      {dqn_stats['count']} orders")
print(f"Fallback assigned: {greedy_stats['count']} orders")
print(f"Total assigned:    {dqn_stats['count'] + greedy_stats['count']} orders")
print(f"Total orders:      {len(env.get_all_order_ids())} orders")

print(f"\nDQN effectiveness: {dqn_stats['count'] / len(env.get_all_order_ids()) * 100:.1f}%")
print(f"Fallback usage:    {greedy_stats['count'] / len(env.get_all_order_ids()) * 100:.1f}%")

if dqn_stats['orders']:
    print(f"\nOrders assigned by DQN (first 10):")
    for oid in list(set(dqn_stats['orders']))[:10]:
        print(f"  - {oid}")
    if len(set(dqn_stats['orders'])) > 10:
        print(f"  ... and {len(set(dqn_stats['orders'])) - 10} more")

if greedy_stats['orders']:
    print(f"\nOrders assigned by FALLBACK (first 10):")
    for oid in greedy_stats['orders'][:10]:
        print(f"  - {oid}")
    if len(greedy_stats['orders']) > 10:
        print(f"  ... and {len(greedy_stats['orders']) - 10} more")

# Restore original functions
Ne3Na3_solver_69.dqn_assignment = original_dqn
Ne3Na3_solver_69.greedy_assignment = original_greedy

# Validate result
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)
is_valid, msg, details = env.validate_solution_complete(result)
print(f"Valid: {is_valid}")

fulfillment = env.get_solution_fulfillment_summary(result)
fulfilled = fulfillment.get('fully_fulfilled_orders', 0)
total = fulfillment['total_orders']
print(f"Fulfillment: {fulfilled}/{total} ({fulfilled/total*100:.1f}%)")

if is_valid:
    cost = env.calculate_solution_cost(result)
    print(f"Cost: ${cost:,.2f}")
