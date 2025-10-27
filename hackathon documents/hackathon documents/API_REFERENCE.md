# Robin Logistics Environment - API Reference

Complete API documentation for contestants using the Robin Logistics Environment.

## Core Environment Interface

The `LogisticsEnvironment` class provides the main interface for contestants to interact with the logistics system.

### Initialization

```python
from robin_logistics import LogisticsEnvironment
from solver import my_solver
env = LogisticsEnvironment()
```

### Solver Management

```python
# Set your solver function
env.set_solver(my_solver)

# Launch dashboard with solver
env.launch_dashboard()

# Run headless with solver
results = env.run_headless("run_001")
```
## Data Models

### Warehouse

```python
warehouse.id                    # Unique identifier
warehouse.location             # Node object
warehouse.inventory            # Dict[sku_id, quantity]
warehouse.vehicles             # List of Vehicle objects
warehouse.pickup_items(sku, quantity)  # Method to remove items
```

### Vehicle

```python
vehicle.id                      # Unique identifier
vehicle.type                    # Type/category of the vehicle
vehicle.home_warehouse_id      # ID of the home warehouse
vehicle.capacity_weight        # Maximum weight capacity in kg
vehicle.capacity_volume        # Maximum volume capacity in m³
vehicle.max_distance           # Maximum route distance in km
vehicle.cost_per_km            # Cost per kilometer
vehicle.fixed_cost             # Fixed operational cost
```

### Order

```python
order.id                        # Unique identifier
order.destination              # Node object representing delivery location
order.requested_items          # Dict[sku_id, quantity]
```

### SKU

```python
sku.id                         # Unique identifier
sku.weight                     # Weight per unit in kg
sku.volume                     # Volume per unit in m³
```

### Node

```python
node.id                        # Unique identifier (integer)
node.lat                       # Latitude coordinate
node.lon                       # Longitude coordinate
```

## Data Access

### Entity Collections

```python
# Entity collections (dicts - access by ID)
env.warehouses          # Dict of Warehouse objects
env.orders              # Dict of Order objects
env.skus                # Dict of SKU objects
env.nodes               # Dict of Node objects

# Collection methods (lists - iterate directly)
env.get_all_vehicles()          # Returns list of Vehicle objects
env.get_all_order_ids()         # Returns list of order IDs
env.get_available_vehicles()    # Returns list of vehicle IDs
```

### Data Structure Usage

**Entity Collections (Dicts):**
- **Access by ID**: `env.warehouses['WH-1']`, `env.orders['ORDER-001']`
- **Iterate with keys/values**: `for id, warehouse in env.warehouses.items()`
- **Check existence**: `if 'WH-1' in env.warehouses`
- **Get count**: `len(env.warehouses)`

**Collection Methods (Lists):**
- **Access by index**: `env.get_all_vehicles()[0]`
- **Iterate directly**: `for vehicle in env.get_all_vehicles()`
- **Check existence**: `if vehicle in env.get_all_vehicles()`
- **Get count**: `len(env.get_all_vehicles())`

### Entity Retrieval

```python
# Get specific entities
env.get_warehouse_by_id(warehouse_id) 
env.get_vehicle_by_id(vehicle_id)
env.orders[order_id]
env.skus[sku_id]
env.nodes[node_id]
```

### Inventory and Capacity

```python
# Warehouse inventory
env.get_warehouse_inventory(warehouse_id)

# Vehicle capacity
env.get_vehicle_remaining_capacity(vehicle_id)  # Returns (weight, volume) tuple
env.get_vehicle_current_capacity(vehicle_id)    # Returns (weight, volume) tuple
env.get_vehicle_current_load(vehicle_id)        # Returns dict of loaded SKUs

# Order requirements
env.get_order_requirements(order_id)
```

### Location and Network

```python
# Get road network data
env.get_road_network_data()

# Distance calculations
env.get_distance(node1_id, node2_id)
env.get_route_distance(route)

# Location information
env.get_order_location(order_id)
env.get_vehicle_home_warehouse(vehicle_id)
```

### Utility Methods

```python
# Find warehouses with specific SKUs
env.get_warehouses_with_sku(sku_id, min_quantity=1)

# Get SKU details
env.get_sku_details(sku_id)

# Get order fulfillment status
env.get_order_fulfillment_status(order_id)
```

## Operations

### Pickup and Delivery

```python
# Pickup from warehouse
success = env.pickup_sku_from_warehouse(
    vehicle_id, warehouse_id, sku_id, quantity
)

# Deliver to order
success = env.deliver_sku_to_order(
    vehicle_id, order_id, sku_id, quantity
)

# Unload to warehouse
success = env.unload_sku_to_warehouse(
    vehicle_id, warehouse_id, sku_id, quantity
)
```

### Route Execution

```python
# Execute sequential route (preferred)
success, message = env.execute_route_sequential(vehicle_id, steps)

# Execute complete solution (routes must include steps)
success, message = env.execute_solution(solution)
```

## Validation

### Route Validation

```python
# Validate sequential route
is_valid, message = env.validator.validate_route_steps(vehicle_id, steps)
```

### Solution Validation

```python
# Validate solution business logic
is_valid, message = env.validate_solution_business_logic(solution)

# Comprehensive solution validation
is_valid, message = env.validate_solution_complete(solution)
```

## Metrics and Analysis

### Cost and Performance

```python
# Calculate solution cost
cost = env.calculate_solution_cost(solution)

# Get cost breakdown (fixed vs variable costs)
cost_breakdown = env.metrics_calculator.calculate_cost_breakdown(solution)
# Returns: {'fixed_cost_total': float, 'variable_cost_total': float, 'total_cost': float}

# Get solution statistics
stats = env.get_solution_statistics(solution)

# Get fulfillment summary
fulfillment = env.get_solution_fulfillment_summary(solution)
```

## Scenario Management

### Problem Generation

```python
# Generate new scenarios
env.generate_new_scenario(seed=42)
env.generate_scenario_from_config(config_dict)

# Random seed management
env.set_random_seed(42)
current_seed = env.get_current_seed()
```



## Solution Format

Your solver must return a solution in this step-based format:

```python
solution = {
    'routes': [
        {
            'vehicle_id': 'vehicle_1',
            'steps': [                # Required: List of step objects
                {
                    'node_id': 1,    # Node ID where this step occurs
                    'pickups': [      # Pickup operations at this node
                        {'warehouse_id': 'WH-1', 'sku_id': 'SKU-001', 'quantity': 5}
                    ],
                    'deliveries': [], # Delivery operations at this node
                    'unloads': []     # Unload operations at this node
                },
                {
                    'node_id': 2,    # Order destination node
                    'pickups': [],
                    'deliveries': [   # Delivery operations at this node
                        {'order_id': 'ORDER-001', 'sku_id': 'SKU-001', 'quantity': 3}
                    ],
                    'unloads': []
                },
                {
                    'node_id': 3,    # Additional delivery node
                    'pickups': [],
                    'deliveries': [   # More delivery operations
                        {'order_id': 'ORDER-001', 'sku_id': 'SKU-001', 'quantity': 2}
                    ],
                    'unloads': []
                },
                {
                    'node_id': 1,    # Return to home warehouse
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                }
            ]
        }
    ]
}
```

**Important**: 
- The `steps` array is **required** for all routes
- Each step must have a valid `node_id`
- Operations (pickups, deliveries, unloads) are bound to specific nodes
- The route must start and end at the vehicle's home warehouse
- Legacy `route` array and aggregated operations are no longer supported

## Route Requirements

### Physical Constraints

- **Start/End**: Route must begin and end at the vehicle's home warehouse node
- **Connectivity**: Each consecutive pair of nodes must be connected
- **Sequential Execution**: Operations at a step must match the step's node (warehouse/destination)

### Business Logic

- **Vehicle Assignment**: Each vehicle can only be assigned to one route
- **Capacity Limits**: Enforced at each step
- **Inventory Availability**: Checked at the warehouse step
- **Order Fulfillment**: Deliveries only at the order's destination node

## Error Handling

### Return Values

Most operations return `(success, message)` tuples:

```python
success, message = env.execute_route(...)

if not success:
    print(f"Operation failed: {message}")
    # Handle error appropriately
```

Some operations return boolean values:

```python
success = env.pickup_sku_from_warehouse(...)

if not success:
    print("Pickup operation failed")
    # Handle error appropriately
```

### Common Error Scenarios

- **Insufficient Inventory**: Warehouse doesn't have required items
- **Capacity Exceeded**: Vehicle cannot carry requested load
- **Invalid Route**: Route doesn't follow road network
- **Wrong Start/End**: Route doesn't start/end at home warehouse
- **Invalid Operations**: Pickup/delivery operations don't match route

## Performance Considerations

- Use `get_road_network_data()` for pathfinding algorithms
- Leverage pre-computed distances with `get_distance()`
- Batch operations when possible
- Validate routes before execution to avoid rollbacks
- Use `get_warehouses_with_sku()` to find warehouses with specific items

## Method Reference

### Core Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `__init__()` | - | Initialize environment |
| `set_solver(func)` | - | Set solver function |
| `launch_dashboard()` | - | Launch interactive dashboard |
| `run_headless(run_id)` | str | Run solver headlessly |

### Data Access Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_all_vehicles()` | List[Vehicle] | Get all vehicles |
| `get_vehicle_by_id(id)` | Vehicle | Get specific vehicle |
| `get_warehouse_by_id(id)` | Warehouse | Get specific warehouse |
| `get_all_order_ids()` | List[str] | Get all order IDs |
| `get_available_vehicles()` | List[str] | Get available vehicle IDs |

### Capacity Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_vehicle_current_load(id)` | Dict | Get current vehicle load |
| `get_vehicle_current_capacity(id)` | Tuple[float, float] | Get current weight/volume usage |
| `get_vehicle_remaining_capacity(id)` | Tuple[float, float] | Get remaining capacity |

### Network Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_distance(node1, node2)` | Optional[float] | Get direct distance |
| `get_route_distance(route)` | float | Get total route distance |
| `get_road_network_data()` | Dict | Get complete network data |

### Operation Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `pickup_sku_from_warehouse(...)` | bool | Pick up SKU from warehouse |
| `deliver_sku_to_order(...)` | bool | Deliver SKU to order |
| `unload_sku_to_warehouse(...)` | bool | Unload SKU to warehouse |
| `execute_route(...)` | Tuple[bool, str] | Execute single route |
| `execute_solution(...)` | Tuple[bool, str] | Execute complete solution |

### Validation Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `validate_route_physical(route)` | Tuple[bool, str] | Validate route connectivity |
| `validate_single_route(...)` | Tuple[bool, str] | Validate complete route |
| `validate_route_feasibility(...)` | Tuple[bool, str] | Validate route operations |
| `validate_solution_business_logic(...)` | Tuple[bool, str] | Validate business rules |
| `validate_solution_complete(...)` | Tuple[bool, str] | Comprehensive validation |

### Metrics Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `calculate_solution_cost(solution)` | float | Calculate total cost |
| `metrics_calculator.calculate_cost_breakdown(solution)` | Dict | Get fixed/variable cost breakdown |
| `get_solution_statistics(solution)` | Dict | Get detailed statistics |
| `get_solution_fulfillment_summary(solution)` | Dict | Get fulfillment summary |

### Utility Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_warehouse_inventory(id)` | Dict | Get warehouse inventory |
| `get_order_requirements(id)` | Dict | Get order requirements |
| `get_warehouses_with_sku(...)` | List[str] | Find warehouses with SKU |
| `get_sku_details(id)` | Optional[Dict] | Get SKU specifications |
| `get_order_location(id)` | int | Get order delivery location |
| `get_vehicle_home_warehouse(id)` | int | Get vehicle home warehouse |
| `get_order_fulfillment_status(id)` | Dict | Get order fulfillment status |
