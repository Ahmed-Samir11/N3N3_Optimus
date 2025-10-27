# Cost Optimization Strategy for Solver 84

## Current Performance Analysis

| Scenario | Fulfillment | Cost    | Distance | Vehicles | Fixed Cost | Variable Cost | Runtime |
|----------|-------------|---------|----------|----------|------------|---------------|---------|
| 1        | 100%        | $935    | 17.4 km  | 3        | $900       | $35           | 6.1s    |
| 2        | 100%        | $1,739  | 100.8 km | 4        | $1,525     | $214          | 9.4s    |
| 3        | **0%**      | $0      | 0 km     | 0        | $0         | $0            | 6.1s    |
| 4        | 100%        | $3,539  | 341.5 km | 6        | $2,775     | $764          | 29.4s   |
| 5        | 100%        | $4,886  | 505.2 km | 7        | $3,650     | $1,236        | 154.1s  |
| 6        | 97%         | $5,869  | 547.4 km | 7        | $4,550     | $1,319        | 60.7s   |

### Key Insights:
- **Scenario 3 is BROKEN** - 0% fulfillment (highest priority fix)
- **Fixed costs dominate** - 65-88% of total cost
- **Scenario 5 is TOO SLOW** - 154 seconds (could timeout on harder scenarios)
- **Scenario 6 missing 1 order** - easy fulfillment win

## 🎯 Optimization Strategy (Priority Order)

### Priority 1: FIX SCENARIO 3 (0% → 100%)
**Issue**: Complete failure  
**Impact**: CATASTROPHIC for competition scoring  
**Action**: Debug why solver crashes/fails on scenario 3
```bash
# Run this to debug:
python debug_scenario_3.py
```
**Expected Impact**: +100 percentage points fulfillment

---

### Priority 2: REDUCE VEHICLE COUNT (Save $400-800 per scenario)
**Issue**: Using 6-7 vehicles per scenario  
**Target**: Reduce by 1-2 vehicles  
**Method**: Better order packing

**Implementation**:
```python
# In build_routes_from_assignments(), before creating routes:

def consolidate_vehicles(env, order_assignments, vehicle_states):
    """
    Try to merge orders from multiple vehicles into fewer vehicles
    """
    # Get current vehicle loads
    vehicle_loads = {}
    for vid, vstate in vehicle_states.items():
        if not vstate['orders']:
            continue
        
        total_weight = 0
        total_volume = 0
        for order_id in vstate['orders']:
            reqs = env.get_order_requirements(order_id)
            for sku_id, qty in reqs.items():
                sku = env.skus[sku_id]
                total_weight += sku.weight * qty
                total_volume += sku.volume * qty
        
        vehicle = env.get_vehicle_by_id(vid)
        vehicle_loads[vid] = {
            'weight': total_weight,
            'volume': total_volume,
            'weight_cap': vehicle.weight_capacity,
            'volume_cap': vehicle.volume_capacity,
            'weight_util': total_weight / vehicle.weight_capacity,
            'volume_util': total_volume / vehicle.volume_capacity,
            'orders': vstate['orders'].copy()
        }
    
    # Find underutilized vehicles (< 70% capacity)
    underutilized = [vid for vid, load in vehicle_loads.items() 
                     if load['weight_util'] < 0.7 and load['volume_util'] < 0.7]
    
    # Try to merge underutilized vehicles
    merged_count = 0
    for vid1 in underutilized:
        if vid1 not in vehicle_loads:
            continue
        
        for vid2 in underutilized:
            if vid1 == vid2 or vid2 not in vehicle_loads:
                continue
            
            # Can we fit vid2's orders into vid1?
            combined_weight = vehicle_loads[vid1]['weight'] + vehicle_loads[vid2]['weight']
            combined_volume = vehicle_loads[vid1]['volume'] + vehicle_loads[vid2]['volume']
            
            if (combined_weight <= vehicle_loads[vid1]['weight_cap'] and
                combined_volume <= vehicle_loads[vid1]['volume_cap']):
                # Merge!
                vehicle_loads[vid1]['orders'].extend(vehicle_loads[vid2]['orders'])
                vehicle_loads[vid1]['weight'] = combined_weight
                vehicle_loads[vid1]['volume'] = combined_volume
                
                # Update assignments
                for order_id in vehicle_loads[vid2]['orders']:
                    order_assignments[vid1][order_id] = order_assignments[vid2][order_id]
                    del order_assignments[vid2][order_id]
                
                del vehicle_loads[vid2]
                merged_count += 1
                print(f"  Merged {vid2} into {vid1}")
                break
    
    return order_assignments, merged_count
```

**Expected Savings**: 1-2 vehicles per scenario = **$400-800**

---

### Priority 3: OPTIMIZE ROUTES WITH 2-OPT (Save 10-20% distance)
**Issue**: Nearest-neighbor TSP is suboptimal  
**Target**: 10-20% distance reduction  
**Method**: 2-opt local search

**Implementation**:
```python
# Add to build_routes_from_assignments(), after TSP ordering:

def apply_2opt(nodes, env):
    """2-opt optimization on delivery sequence"""
    if len(nodes) <= 3:
        return nodes
    
    route = nodes[:]
    improved = True
    iterations = 0
    max_iter = 50  # Limit to prevent timeout
    
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        
        for i in range(len(route) - 2):
            for j in range(i + 2, len(route)):
                # Calculate distance change
                # Current: ...→route[i]→route[i+1]...route[j-1]→route[j]→...
                # New:     ...→route[i]→route[j-1]...(reversed)...route[i+1]→route[j]→...
                
                old_dist = (
                    (env.get_distance(route[i], route[i+1]) or 0) +
                    (env.get_distance(route[j-1], route[j]) or 0)
                )
                
                new_dist = (
                    (env.get_distance(route[i], route[j-1]) or 0) +
                    (env.get_distance(route[i+1], route[j]) or 0)
                )
                
                if new_dist < old_dist:
                    # Reverse segment [i+1 : j]
                    route[i+1:j] = reversed(route[i+1:j])
                    improved = True
    
    return route

# Usage:
# delivery_nodes = apply_2opt(delivery_nodes, env)
```

**Expected Savings**: 50-100 km per scenario = **$100-200**

---

### Priority 4: FIX SCENARIO 6 UNFULFILLED ORDER (97% → 100%)
**Issue**: 1 order unfulfilled (34/35)  
**Target**: 100% fulfillment  
**Method**: Ensure greedy fallback catches all orders

**Implementation**:
```python
# At end of DQN assignment, add aggressive fallback:

# After DQN assignment
assigned_orders = set()
for vid, vstate in vehicle_states.items():
    assigned_orders.update(vstate['orders'])

unassigned = set(all_order_ids) - assigned_orders

# AGGRESSIVE FALLBACK: Try to fit unassigned orders ANYWHERE
for order_id in unassigned:
    reqs = env.get_order_requirements(order_id)
    order_weight = sum(env.skus[sku].weight * qty for sku, qty in reqs.items())
    order_volume = sum(env.skus[sku].volume * qty for sku, qty in reqs.items())
    
    # Try EVERY vehicle, even if capacity is tight
    best_vehicle = None
    min_overage = float('inf')
    
    for vehicle in env.get_all_vehicles():
        rem_weight, rem_volume = env.get_vehicle_remaining_capacity(vehicle.id)
        weight_overage = max(0, order_weight - rem_weight)
        volume_overage = max(0, order_volume - rem_volume)
        total_overage = weight_overage + volume_overage
        
        if total_overage < min_overage:
            min_overage = total_overage
            best_vehicle = vehicle
    
    # Assign to best vehicle (even if slightly over capacity - solver might work it out)
    if best_vehicle and min_overage < 100:  # Some tolerance
        # Assign order to vehicle
        pass
```

**Expected Savings**: Penalty avoidance for unfulfilled order

---

### Priority 5: SPEED UP SCENARIO 5 (154s → < 60s)
**Issue**: Solver taking too long  
**Target**: Under 60 seconds  
**Method**: Early termination, simpler heuristics

**Implementation**:
```python
# Add timeout mechanism:
import time

def solver(env):
    start_time = time.time()
    TIMEOUT_SECONDS = 50  # Leave buffer before 30min limit
    
    # During optimization loops:
    if time.time() - start_time > TIMEOUT_SECONDS:
        print(f"Timeout reached, returning best solution so far...")
        break
```

**Expected Impact**: Prevents timeouts on harder scenarios

---

## 📈 Expected Total Cost Savings

| Optimization           | Cost Savings | Difficulty | Priority |
|------------------------|--------------|------------|----------|
| Fix Scenario 3         | ∞ (0%→100%)  | Medium     | 🔴 P1    |
| Reduce vehicles (-2)   | $400-800     | Medium     | 🟡 P2    |
| 2-opt routing          | $100-200     | Easy       | 🟢 P3    |
| Fix Scenario 6 order   | Penalty avoid| Easy       | 🟢 P4    |
| Speed up Scenario 5    | Reliability  | Easy       | 🟢 P5    |
| **TOTAL**              | **$500-1000**| -          | -        |

## 🚀 Implementation Order

1. **Debug Scenario 3** - Run `debug_scenario_3.py` to understand failure
2. **Add 2-opt** - Quick win, easy to implement
3. **Vehicle consolidation** - Bigger savings, moderate effort
4. **Aggressive fallback** - Catch last unfulfilled order
5. **Add timeout** - Prevent future timeouts

## 📝 Testing Plan

After each change:
```bash
python compare_solvers.py  # Compare solver_84 vs solver_85 (with improvements)
```

Track metrics:
- Fulfillment % (must stay 100%)
- Total cost (target: reduce by $500-1000)
- Runtime (target: all scenarios < 60s)
- Vehicle count (target: reduce by 1-2 per scenario)
