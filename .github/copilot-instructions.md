# Robin Logistics Solver - Copilot Instructions

## Project Overview

This is a competitive logistics optimization project for the **Beltone 2nd AI Hackathon (Powered by Robin)**, solving Multi-Warehouse Vehicle Routing Problems (MWVRP) over Cairo's road network. The codebase contains multiple iterative solver implementations (`Ne3Na3_solver_1.py` through `Ne3Na3_solver_7.py`) exploring different optimization strategies.

### Core Domain: Multi-Warehouse Vehicle Routing
- **Problem**: 332k node directional road network, 50 orders, 12 vehicles, 3 SKUs, 2 warehouses
- **Objective**: **FULFILLMENT FIRST**, then minimize cost (competition scoring heavily penalizes unfulfilled orders)
- **Constraints**: Vehicle capacity (weight/volume), warehouse inventory, road network connectivity, 30-minute solver runtime limit
- **Key Challenge**: Multi-warehouse order splitting when single warehouses lack inventory

### Competition Scoring (Critical)
```
Scenario Score = Your Cost + Benchmark Cost × (100 - Your Fulfillment %)
```
- **Lower is better** - missing fulfillment is heavily penalized
- Dynamic ranking across multiple private scenarios (20 pts for 1st, 19 for 2nd, etc.)
- **Strategy**: Prioritize 100% fulfillment, THEN optimize cost

## Architecture & Entry Points

### Solver Contract (MANDATORY)
```python
def solver(env):  # DO NOT initialize environment inside
    # Your algorithm here
    return {"routes": [...]}
```
- **Input**: `LogisticsEnvironment` instance (from `robin-logistics-env` package)
- **Output**: Solution dict with routes
- **Rules**: NO CACHING between runs, NO environment manipulation, 30-min runtime limit
- **Submission**: Comment out `if __name__ == '__main__'` block, name file `Ne3Na3_solver_N.py`

### Solution Format - Critical Structure
Routes MUST include intermediate path nodes (not just pickup/delivery stops):
```python
{
    "routes": [
        {
            "vehicle_id": "V-1",
            "steps": [  # Sequential nodes including intermediate road segments
                {"node_id": 1, "pickups": [], "deliveries": [], "unloads": []},
                {"node_id": 5, "pickups": [{"warehouse_id": "WH-1", "sku_id": "Light_Item", "quantity": 30}], "deliveries": [], "unloads": []},
                {"node_id": 7, "pickups": [], "deliveries": [], "unloads": []},  # Intermediate node
                {"node_id": 10, "pickups": [], "deliveries": [{"order_id": "ORD-1", "sku_id": "Light_Item", "quantity": 30}], "unloads": []},
                {"node_id": 1, "pickups": [], "deliveries": [], "unloads": []}  # Return home
            ]
        }
    ]
}
```

## Critical Patterns From This Codebase

### 1. Distance & Pathfinding (MANDATORY)
**DO**: Use Dijkstra with per-run caching (cleared on each `solver()` call)
```python
# Pattern from Ne3Na3_solver_4.py, Ne3Na3_solver_6.py
distance_cache = {}  # Per-run only - cleared between solver calls
path_cache = {}      # Stores (distance, path_nodes) tuples

def dijkstra_with_path(env, start, end):
    # Returns (distance, [node_list]) including intermediate nodes
    # This pattern is in ALL working solvers
```

**DON'T**: Use direct distance lookups without path expansion - routes will fail validation

### 2. Warehouse-Vehicle Architecture
- Vehicles belong to specific warehouses (`vehicle.home_warehouse_id`)
- Each warehouse has `inventory` dict and `vehicles` list
- Access via: `env.warehouses`, `env.get_all_vehicles()`, `env.orders`, `env.skus`
- Road network: `env.get_road_network_data()` returns `{"adjacency_list": {}, "edges": []}`

### 3. Multi-Warehouse Order Allocation Strategy
From `Ne3Na3_solver_2.py` (most sophisticated approach):
1. **Split orders across warehouses** when single warehouse lacks full inventory
2. **Track inventory depletion** as orders are assigned
3. **Distance-weighted allocation**: prefer closer warehouses for partial fulfillment
4. Result: `shipments[warehouse_id][order_id][sku_id] = quantity`

### 4. Capacity Management
```python
# Check before assignment (pattern from Ne3Na3_solver_1.py)
rem_weight, rem_volume = env.get_vehicle_remaining_capacity(vehicle.id)
sku = env.skus[sku_id]
needed_weight = sku.weight * quantity
needed_volume = sku.volume * quantity
if needed_weight <= rem_weight and needed_volume <= rem_volume:
    # Assign order
```

### 5. Route Building - Step Sequence Construction
**Critical**: Use `build_steps_with_path()` pattern from `Ne3Na3_solver_7.py`:
```python
def build_steps_with_path(env, node_sequence, home_node):
    """
    Expands compact (pickup/delivery) sequence into full path with intermediate nodes.
    node_sequence: [(node_id, {"pickups": [...], "deliveries": [...]}), ...]
    """
    steps = [{"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []}]
    current = home_node
    for node_id, payload in node_sequence:
        path, _ = dijkstra_shortest_path(env, current, node_id)
        # Insert intermediate nodes before delivery node
        for intermediate in path[1:]:
            steps.append(...)
        current = node_id
    # Path back to home
    return steps
```

## Solver Evolution Strategy (Team Ne3Na3's Progression)

1. **Solver 1**: Basic greedy + BFS pathfinding → Baseline
2. **Solver 2**: Multi-warehouse splitting + Dijkstra + 2-opt → Better fulfillment
3. **Solver 4**: Per-run caching + connectivity checks → Faster
4. **Solver 6**: Two-phase (allocation + LNS) → Higher optimization
5. **Solver 7**: Memetic algorithm (population-based) → Strong baseline
6. **Solver 8**: ML-enhanced memetic (K-means clustering, multi-criteria scoring, adaptive search) → **Current best**

## Development Workflows

### Testing Your Solver
```powershell
# Headless mode (fast validation) - PREFERRED for development
python -c "from robin_logistics import LogisticsEnvironment; from Ne3Na3_solver_7 import solver; env = LogisticsEnvironment(); result = solver(env); print(env.execute_solution(result))"

# Dashboard mode (visual debugging)
# In hackathon documents/hackathon documents/:
python run_dashboard.py  # Edit to import your solver
```

### Test Scripts Available
```powershell
# Test road network structure
python "hackathon documents/hackathon documents/test_network.py"

# Debug distance calculation
python "hackathon documents/hackathon documents/test_distance.py"

# Validate solver output format
python "hackathon documents/hackathon documents/test_solver.py"

# Run headless with metrics
python "hackathon documents/hackathon documents/test_headless.py"

# Debug environment state
python "hackathon documents/hackathon documents/test_solver_debug.py"
```

### Common Validation Errors
1. **"Route must start at vehicle home"** → Check `env.get_vehicle_home_warehouse(vehicle_id)`
2. **"Insufficient inventory"** → Verify warehouse has SKUs before pickup step
3. **"Capacity exceeded"** → Calculate weight/volume BEFORE adding deliveries
4. **"Invalid path"** → Missing intermediate nodes between stops
5. **"Running time exceeded"** → Optimize algorithm to finish within 30 minutes

### Evaluating Solution Quality
```python
# Check fulfillment (MOST IMPORTANT)
fulfillment = env.get_solution_fulfillment_summary(solution)
fulfilled = fulfillment.get("fully_fulfilled_orders", 0)
total_orders = len(env.get_all_order_ids())
fulfillment_pct = 100 * fulfilled / total_orders

# Check cost (secondary to fulfillment)
cost = env.calculate_solution_cost(solution)

# Competition scoring approximation (needs benchmark cost)
# scenario_score = cost + benchmark_cost * (100 - fulfillment_pct)
print(f"Fulfillment: {fulfilled}/{total_orders} ({fulfillment_pct:.1f}%)")
print(f"Cost: ${cost:,.0f}")
```

### Python Environment Setup
```powershell
# Activate virtual environment (already configured at ./optimus/)
& c:/Users/ahmed/OneDrive/Desktop/Beltone/N3N3_Optimus/optimus/Scripts/Activate.ps1
pip install robin-logistics-env
```

## Project-Specific Conventions

### File Naming
- **`Ne3Na3_solver_N.py`**: Competition submissions (numbered iterations)
- **`hackathon documents/hackathon documents/`**: Reference materials, test scripts
- Solvers are **standalone** - no cross-file imports between solver versions

### Data Structure Access Patterns
```python
# Entity collections are DICTS (not lists) - iterate with .items()
for wh_id, warehouse in env.warehouses.items():
    ...

# Vehicle collections require method call
for vehicle in env.get_all_vehicles():
    ...

# Orders accessed by ID
order = env.orders[order_id]
requirements = env.get_order_requirements(order_id)  # Returns dict[sku_id, quantity]
```

### No Global Caching Between Runs (CRITICAL RULE)
Competition rules **PROHIBIT** cross-run state. Per-run caching is ALLOWED:
```python
def solver(env):
    distance_cache = {}  # Fresh cache per solver call - ALLOWED
    path_cache = {}      # Cleared when function returns - ALLOWED
    # Use cache within this run only
    # Do NOT persist cache to global/file scope
```

## Key API Methods

### Essential Operations
- `env.get_distance(node1, node2)` → Direct edge distance or None
- `env.get_road_network_data()` → `{"adjacency_list": {node: [neighbors]}, "edges": [...]}`
- `env.get_order_requirements(order_id)` → `{sku_id: quantity}`
- `env.get_warehouse_inventory(warehouse_id)` → `{sku_id: quantity}`
- `env.validate_solution_complete(solution)` → `(bool, str)` - Use before execute

### Validation & Execution
```python
# Always validate before running
is_valid, msg = env.validate_solution_complete(solution)
if is_valid:
    success, exec_msg = env.execute_solution(solution)
    cost = env.calculate_solution_cost(solution)
    fulfillment = env.get_solution_fulfillment_summary(solution)
```

## Optimization Techniques in This Codebase

1. **Greedy Construction** (Solver 1-4): Assign orders one-by-one to available vehicles
2. **Nearest Neighbor + 2-opt** (Solver 2-3): TSP-style route ordering
3. **Large Neighborhood Search (LNS)** (Solver 6): Destroy/repair operators
4. **Memetic Algorithm** (Solver 7): Population + crossover + mutation
5. **ML-Enhanced Memetic** (Solver 8): K-means clustering + multi-criteria scoring + adaptive search

### ML/Data-Science Techniques (Solver 8)
- **K-Means Clustering**: Groups orders spatially for efficient routing (15-25% distance reduction)
- **Multi-Criteria Affinity Scoring**: 70% distance + 30% capacity utilization for order-vehicle matching
- **Smart Warehouse Allocation**: Distance-weighted multi-warehouse splitting with inventory tracking
- **Tournament Selection**: Top-3 parent selection in memetic algorithm
- **Early Stopping**: Adaptive convergence detection (stops after 20 iterations without improvement)
- **Fulfillment-First Evaluation**: Prioritizes 100% order fulfillment before optimizing cost

### Optimization Strategy Priorities
Given competition scoring `Score = Cost + Benchmark × (100 - Fulfillment%)`:
1. **First**: Achieve 100% order fulfillment (unfulfilled orders heavily penalized)
2. **Second**: Minimize total cost (fixed + variable costs)
3. **Tradeoff**: Use more vehicles if needed for fulfillment, don't sacrifice orders for cost savings

## Non-Obvious Gotchas

- **Vehicle home != Warehouse node**: Use `env.get_vehicle_home_warehouse()` not `warehouse.location.id`
- **Order destination is Node object**: Access via `order.destination.id` (has `.lat`, `.lon`)
- **SKUs have per-unit weight/volume**: Multiply by quantity when calculating capacity
- **Pickup steps need `warehouse_id`**: `{"warehouse_id": "WH-1", "sku_id": ..., "quantity": ...}`
- **Delivery steps need `order_id`**: `{"order_id": "ORD-1", "sku_id": ..., "quantity": ...}`

## Reference Files

- **API Documentation**: `hackathon documents/hackathon documents/API_REFERENCE.md`
- **Example Solver**: `hackathon documents/hackathon documents/solver.py` (basic BFS implementation)
- **Best Solver**: `Ne3Na3_solver_8.py` (ML-enhanced memetic with K-means clustering, multi-criteria scoring)
- **ML Improvements Doc**: `SOLVER_8_IMPROVEMENTS.md` (detailed explanation of data science enhancements)
