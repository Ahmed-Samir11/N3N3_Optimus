from robin_logistics import LogisticsEnvironment


def solver(env):
    """
    Simplified debug version:
    - Prints diagnostic info to identify why routes were empty.
    - Always produces at least one route if any vehicle & order exist.
    """

    order_ids = env.get_all_order_ids()
    vehicle_ids = env.get_available_vehicles()
    warehouses = list(env.warehouses.keys())

    if not order_ids or not vehicle_ids:
        print("No orders or vehicles available.")
        return {"routes": []}

    # pick first vehicle & first order just to test connectivity
    v_id = vehicle_ids[0]
    o_id = order_ids[0]
    try:
        home_id = env.get_vehicle_home_warehouse(v_id)
    except Exception:
        home_id = None
    try:
        order_node = env.get_order_location(o_id)
    except Exception:
        order_node = None

    # choose any warehouse that has any stock
    wid = None
    for w in warehouses:
        inv = env.get_warehouse_inventory(w) or {}
        if any(q > 0 for q in inv.values()):
            wid = w
            break
    if wid is None and warehouses:
        wid = warehouses[0]

    try:
        wnode = env.get_warehouse_by_id(wid).location.id
    except Exception:
        wnode = None

    # Build a minimal valid route
    if home_id is None or wnode is None or order_node is None:
        print("One of node IDs is None; cannot build route.")
        return {"routes": []}

    # Take first order requirements
    req = env.get_order_requirements(o_id)
    deliveries = [{'order_id': o_id, 'sku_id': s, 'quantity': int(q)} for s, q in req.items()]
    pickups = [{'warehouse_id': wid, 'sku_id': s, 'quantity': int(q)} for s, q in req.items()]

    route = {
        "vehicle_id": v_id,
        "steps": [
            {"node_id": home_id, "pickups": [], "deliveries": [], "unloads": []},
            {"node_id": wnode, "pickups": pickups, "deliveries": [], "unloads": []},
            {"node_id": order_node, "pickups": [], "deliveries": deliveries, "unloads": []},
            {"node_id": home_id, "pickups": [], "deliveries": [], "unloads": []},
        ],
    }

    solution = {"routes": [route]}
    return solution

if __name__ == "__main__":
    env = LogisticsEnvironment()
    solution = solver(env)
    print(f"Built {len(solution.get('routes', []))} routes.")