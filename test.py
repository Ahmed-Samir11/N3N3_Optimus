from robin_logistics import LogisticsEnvironment

env = LogisticsEnvironment()

warehouses = env.warehouses

print(warehouses.items())

print(env.get_warehouse_inventory('WH-1'))

print(env.get_order_requirements('ORD-1'))

road_network = env.get_road_network_data()

wh_obj = env.get_warehouse_by_id('WH-1')

print(env.get_order_location('ORD-1'))

wh_obj.location