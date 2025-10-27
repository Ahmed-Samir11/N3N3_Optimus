"""
ROBIN LOGISTICS ENVIRONMENT - QUICK REFERENCE GUIDE
====================================================

PROBLEM TYPE: Multi-Warehouse Vehicle Routing Problem (MWVRP)
SCALE: 7,522 nodes, 16,357 edges, 50 orders, 12 vehicles, 2 warehouses, 3 SKUs

═══════════════════════════════════════════════════════════════════════════════
1. DECISION VARIABLES (What We Control)
═══════════════════════════════════════════════════════════════════════════════

PRIMARY DECISIONS:
  ✓ Which vehicles to use (binary selection)
  ✓ Which orders assigned to which vehicle
  ✓ Which warehouse fulfills each order
  ✓ Route sequencing (order of stops)
  ✓ Path selection (which roads to take)

DERIVED DECISIONS:
  • Pickup quantities per warehouse
  • Delivery sequence optimization
  • Multi-warehouse order splitting
  • Unload operations (if needed)

═══════════════════════════════════════════════════════════════════════════════
2. ENVIRONMENT STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

ROAD NETWORK:
  • Nodes: ~7,500 (Cairo road network)
  • Edges: ~16,000 (directional roads)
  • Avg connectivity: 2.17 edges/node
  • Pathfinding: Dijkstra required

WAREHOUSES (2):
  • WH-1: Node 29542602 (30.092540, 31.315476)
  • WH-2: Node 31272168 (30.110570, 31.369969)
  • Each has: ~110-115 units per SKU (330+ total items)
  • Each manages: 6 vehicles (2 Heavy, 2 Medium, 2 Light)

VEHICLES (12 total):
  Type         | Count | Weight  | Volume | Fixed | Var/km | Max Dist | $/Unit
  -------------|-------|---------|--------|-------|--------|----------|--------
  HeavyTruck   |   2   | 5000kg  | 20m³   | $1200 | $1.50  | 200km    | $0.239
  MediumTruck  |   4   | 1600kg  | 6m³    | $625  | $1.25  | 150km    | $0.389
  LightVan     |   6   | 800kg   | 3m³    | $300  | $1.00  | 100km    | $0.374
  
  KEY INSIGHT: HeavyTruck has BEST cost efficiency ($0.239/unit)
               Use HeavyTrucks first, then LightVans, then MediumTrucks

ORDERS (50):
  • Avg weight: 218kg/order
  • Avg volume: 0.87m³/order
  • Total demand: 10,905kg, 43.62m³
  • Delivery locations: Spread across Cairo network

SKUs (3):
  Light_Item:  5kg, 0.02m³
  Medium_Item: 15kg, 0.06m³
  Heavy_Item:  30kg, 0.12m³

═══════════════════════════════════════════════════════════════════════════════
3. CONSTRAINTS (What We Must Satisfy)
═══════════════════════════════════════════════════════════════════════════════

HARD CONSTRAINTS (Validation fails if violated):
  ✓ Route start/end at vehicle home warehouse
  ✓ Capacity: weight AND volume respected at every step
  ✓ Inventory: can't pickup more than available
  ✓ Connectivity: only travel on existing edges
  ✓ Vehicle uniqueness: each vehicle used max once
  ✓ Pickup before delivery
  ✓ Order integrity: all SKUs together

SOFT CONSTRAINTS (Affect scoring):
  ✓ Fulfillment rate (maximize - heavily penalized if missing)
  ✓ Cost minimization
  ✓ Capacity utilization

OPERATIONAL CONSTRAINTS:
  ✓ 30-minute runtime limit
  ✓ No environment manipulation
  ✓ Valid solution dict format
  ✓ No string-based solutions

═══════════════════════════════════════════════════════════════════════════════
4. SOLUTION FORMAT
═══════════════════════════════════════════════════════════════════════════════

{
  "routes": [
    {
      "vehicle_id": "HeavyTruck_WH-1_1",
      "steps": [
        {"node_id": 29542602, "pickups": [...], "deliveries": [], "unloads": []},
        {"node_id": 7208831013, "pickups": [], "deliveries": [], "unloads": []},
        {"node_id": 6584155087, "pickups": [], "deliveries": [...], "unloads": []},
        ...
        {"node_id": 29542602, "pickups": [], "deliveries": [], "unloads": []}
      ]
    }
  ]
}

CRITICAL REQUIREMENTS:
  ⚠ Include ALL intermediate path nodes (not just pickup/delivery stops)
  ⚠ Pickups need {"warehouse_id": "WH-1", "sku_id": "...", "quantity": ...}
  ⚠ Deliveries need {"order_id": "ORD-1", "sku_id": "...", "quantity": ...}
  ⚠ First step = home warehouse, Last step = return home
  ⚠ No consecutive duplicate nodes (causes 0 distance bug)

═══════════════════════════════════════════════════════════════════════════════
5. SCORING FORMULA (Competition)
═══════════════════════════════════════════════════════════════════════════════

Scenario Score = Your Cost + Benchmark Cost × (100 - Your Fulfillment %)

EXAMPLE:
  Your Cost = $3,500
  Benchmark = $4,000
  Fulfillment = 96% (48/50 orders)
  
  Score = 3,500 + 4,000 × (100 - 96)
        = 3,500 + 4,000 × 4
        = 3,500 + 16,000
        = $19,500 ❌ HUGE PENALTY!

  With 100% fulfillment:
  Score = 3,500 + 4,000 × 0 = $3,500 ✅

KEY TAKEAWAY: Missing 2 orders costs $16,000!
              ALWAYS prioritize 100% fulfillment first!

═══════════════════════════════════════════════════════════════════════════════
6. OPTIMIZATION STRATEGY
═══════════════════════════════════════════════════════════════════════════════

PROVEN APPROACH (Solver 53/54):

Phase 1: Adaptive Bin Packing (5-8 seconds)
  1. Sort orders by size (large first)
  2. Alternate warehouse allocation (load balancing)
  3. Rank vehicles by cost efficiency (HeavyTruck > LightVan > MediumTruck)
  4. Greedy pack orders into vehicles
  5. Build routes with TSP ordering

Phase 2: 2-Opt Local Search (15-20 seconds)
  1. Optimize delivery sequences per route
  2. Time-bounded: 2 seconds per route
  3. Early stopping if no improvement

Result: 100% fulfillment, $3,200-$3,400 cost, ~10 seconds runtime

═══════════════════════════════════════════════════════════════════════════════
7. COMMON PITFALLS TO AVOID
═══════════════════════════════════════════════════════════════════════════════

❌ Missing intermediate path nodes → Distance = 0, validation fails
❌ Duplicate steps at same node → Environment can't calculate distance
❌ Pickup without warehouse_id → Inventory validation fails
❌ Starting at wrong node → Route validation fails
❌ Exceeding capacity → Hard constraint violation
❌ Sacrificing fulfillment for cost → Massive scoring penalty
❌ Persistent caching across runs → Against competition rules

✅ Use NetworkX for fast pathfinding
✅ Clear caches at start of solver()
✅ Merge pickup into first warehouse step (no duplicates)
✅ Include all intermediate nodes in paths
✅ Prioritize 100% fulfillment over cost savings
✅ Test on multiple scenarios (not just one)

═══════════════════════════════════════════════════════════════════════════════
8. PERFORMANCE BENCHMARKS
═══════════════════════════════════════════════════════════════════════════════

Performance Tiers:
  Baseline:   $4,000-$5,000 (greedy, no optimization, 90% fulfillment)
  Good:       $3,500-$4,000 (bin packing + TSP, 95% fulfillment)
  Excellent:  $3,200-$3,500 (adaptive + 2-opt, 100% fulfillment) ✅
  Best:       $3,000-$3,200 (advanced optimization, 100% fulfillment)

Target: $3,200 with 100% fulfillment in ~10 seconds

═══════════════════════════════════════════════════════════════════════════════
9. KEY API METHODS
═══════════════════════════════════════════════════════════════════════════════

VALIDATION:
  env.validate_solution_complete(solution) → (bool, msg)

EXECUTION:
  env.execute_solution(solution) → (bool, msg)

METRICS:
  env.calculate_solution_cost(solution) → float
  env.get_solution_fulfillment_summary(solution) → dict
    • fully_fulfilled_orders
    • total_orders
    • average_fulfillment_rate
    • total_distance
    • total_cost
    • vehicle_utilization

DATA ACCESS:
  env.warehouses → dict[warehouse_id, Warehouse]
  env.orders → dict[order_id, Order]
  env.skus → dict[sku_id, SKU]
  env.get_all_vehicles() → list[Vehicle]
  env.get_all_order_ids() → list[str]
  env.get_order_requirements(order_id) → dict[sku_id, quantity]
  env.get_warehouse_inventory(wh_id) → dict[sku_id, quantity]
  env.get_road_network_data() → dict['adjacency_list', 'edges']

═══════════════════════════════════════════════════════════════════════════════
10. WINNING FORMULA
═══════════════════════════════════════════════════════════════════════════════

1. FULFILLMENT FIRST: Achieve 100% order fulfillment
2. Cost-efficient vehicles: Use HeavyTrucks (best $/unit)
3. Smart allocation: Size-sorted bin packing, alternating warehouses
4. Route optimization: TSP + 2-opt local search
5. Fast pathfinding: NetworkX with @lru_cache
6. Time management: 5-8s construction, 15-20s optimization
7. Generic approach: Works across all scenarios
8. Proper validation: Include all intermediate path nodes

Expected Result: 100% fulfillment, $3,200-$3,400 cost, ~10s runtime
"""

print(__doc__)
