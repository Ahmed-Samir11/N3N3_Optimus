"""
Quick analysis: Can we fit 50 orders in 2 HeavyTrucks + 1 LightVan?

HeavyTruck capacity: 5000kg / 20m³
LightVan capacity: 800kg / 3m³
Total capacity: 10,800kg / 43m³

Let's check total demand.
"""

from robin_logistics import LogisticsEnvironment

env = LogisticsEnvironment()

total_weight = 0
total_volume = 0

for oid in env.get_all_order_ids():
    req = env.get_order_requirements(oid)
    for sku_id, qty in req.items():
        sku = env.skus[sku_id]
        total_weight += sku.weight * qty
        total_volume += sku.volume * qty

print(f"Total demand: {total_weight}kg / {total_volume}m³")
print(f"\n2 HeavyTrucks: {2*5000}kg / {2*20}m³")
print(f"1 LightVan: 800kg / 3m³")
print(f"Total capacity (2H+1L): {10800}kg / {43}m³")
print(f"\nFits? Weight: {total_weight <= 10800}, Volume: {total_volume <= 43}")

# Check per warehouse
for wh_id in ['WH-1', 'WH-2']:
    wh = env.warehouses[wh_id]
    print(f"\n{wh_id} inventory:")
    for sku_id, qty in wh.inventory.items():
        print(f"  {sku_id}: {qty}")
