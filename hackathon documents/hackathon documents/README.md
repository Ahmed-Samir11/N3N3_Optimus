# Robin Logistics Environment — Contestant Guide

## Install or Upgrade

Install:
```bash
pip install robin-logistics-env
```

Upgrade to latest version:
```bash
pip install --upgrade robin-logistics-env
```

## Available Solvers

- `solver.py` — **my_solver**: Basic BFS-based routing solver (recommended starting point)
- `manual_assignments_dashboard.py` — **manual_assignments_solver_main**: Manual route assignment solver

## Quick Start

### 1. Test Your Setup
```bash
python test_all.py
```

### 2. Basic Solver (Recommended)
```bash
# Run headless
python run_headless.py

# Run dashboard
python run_dashboard.py
```

### 2. Manual Assignments Solver
```bash

python -c "
from robin_logistics import LogisticsEnvironment
from manual_assignments_dashboard import manual_assignments_solver_main
env = LogisticsEnvironment()
env.set_solver(manual_assignments_solver_main)
env.launch_dashboard()
"
```

#### Example ASSIGNMENTS (steps-only)
```python
ASSIGNMENTS = [
    {
        'vehicle_id': 'V-1',
        'steps': [
            {'node_id': 1, 'pickups': [], 'deliveries': [], 'unloads': []},
            {'node_id': 5, 'pickups': [{'warehouse_id': 'WH-1', 'sku_id': 'Light_Item', 'quantity': 30}], 'deliveries': [], 'unloads': []},
            {'node_id': 10, 'pickups': [], 'deliveries': [{'order_id': 'ORD-1', 'sku_id': 'Light_Item', 'quantity': 30}], 'unloads': []},
            {'node_id': 1, 'pickups': [], 'deliveries': [], 'unloads': []},
        ]
    }
]
```

## ⚙️ Configuration & Custom Scenarios

### Build Custom Scenarios
```python
# Example: Create focused test scenario
custom_config = {
    'random_seed': 42,                    # Reproducible results
    'num_orders': 15,                     # Smaller for testing
    'sku_percentages': [50, 30, 20],      # More light items
    
    'distance_control': {
        'radius_km': 10,                  # Compact area
        'density_strategy': 'clustered',   # Orders near warehouses
        'clustering_factor': 0.9,         # Highly clustered
    },
    
    'warehouse_configs': [
        {
            'vehicle_counts': {'LightVan': 2, 'MediumTruck': 1},
            'sku_inventory_percentages': [80, 60, 40]  # WH-1 has more stock
        },
        {
            'vehicle_counts': {'LightVan': 1, 'HeavyTruck': 1},
            'sku_inventory_percentages': [20, 40, 60]  # WH-2 complements WH-1
        }
    ]
}

env.generate_scenario_from_config(custom_config)
```

### Configuration Options
- **Core**: `random_seed`, `num_orders`, `sku_percentages`
- **Geographic**: `radius_km`, `density_strategy` ('uniform'/'clustered'/'ring'), `clustering_factor`
- **Fleet**: `vehicle_counts` per warehouse (LightVan/MediumTruck/HeavyTruck)
- **Inventory**: `sku_inventory_percentages` (% of total demand per SKU)

### Scenario Export/Import
```python
# Export with all configuration
scenario = env.export_scenario()

# Save to disk
import json
with open('scenario.json', 'w') as f:
    json.dump(scenario, f, indent=2)

# Load later
with open('scenario.json', 'r') as f:
    scenario = json.load(f)
env.load_scenario(scenario)

# Access generation config
gen_cfg = env.get_stored_generation_config()
print(f"Used clustering: {gen_cfg['distance_control']['clustering_factor']}")
```

## File Structure

- `solver.py` — Main contestant solver with BFS pathfinding
- `run_headless.py` — Headless execution runner
- `run_dashboard.py` — Dashboard launcher
- `run_with_scenario.py` — Run with saved scenario data
- `manual_assignments_dashboard.py` — Alternative solver approach
- `API_REFERENCE.md` — Complete API documentation

## Solver Entry Points

- **my_solver(env)** — Main solver function in solver.py
- **manual_assignments_solver_main(env)** — Manual assignments solver

## Notes

- The dashboard runner uses `env.launch_dashboard()` for proper integration
- Headless validates, executes, and prints basic statistics
- Both modes use the same centralized validation, execution, and metrics paths (identical logic)
- Baseline reset ensures headless runs do not deplete inventory for dashboard runs
- Reproducibility: using the same `random_seed` yields the same scenario in both modes
- All solvers automatically work in both dashboard and headless modes

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the `contestant` directory
2. **Solver Not Found**: Verify your solver function is named correctly (e.g., `my_solver`)
3. **Dashboard Won't Launch**: Check if port 8501 is available, or use a different port

### State Management

The environment automatically performs a true reset between dashboard runs (restores baseline warehouse inventories and clears delivered items). You can also reset manually:

```python
# Reset all state (inventory, vehicles, orders)
env.reset_all_state()

# Complete reset with new scenario
env.complete_reset(seed=42)

# Reset only vehicle states
env._reset_vehicle_states()
```

**When to use:**
- After running headless mode before dashboard
- When switching between different test scenarios
- If inventory appears depleted from previous runs
- For reproducible testing with specific seeds

### Testing

Run the test suite to verify your setup:
```bash
python test_all.py
```

This will check:
- All imports work correctly
- Environment can be created
- Solver functions can be called
- Basic functionality is working

### Getting Help

- Check `API_REFERENCE.md` for complete API documentation
- Review the solver examples in `solver.py` and `manual_assignments_dashboard.py`
- Ensure you're using the latest version: `pip install --upgrade robin-logistics-env`