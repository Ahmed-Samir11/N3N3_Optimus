"""Test connectivity of problematic nodes"""
from robin_logistics import LogisticsEnvironment

env = LogisticsEnvironment()

# Problematic nodes from test output
problem_nodes = [
    (270423014, 6594748982),   # Run 2
    (8210241011, 272523750),   # Run 2
    (8210241011, 7188563114),  # Run 2
    (268010440, 8010777099),   # Run 3
    (6580098851, 670595492),   # Run 4
]

# Warehouse nodes
wh1_node = env.warehouses['WH-1'].location.id
wh2_node = env.warehouses['WH-2'].location.id

print(f"WH-1 node: {wh1_node}")
print(f"WH-2 node: {wh2_node}")
print()

# Simple Dijkstra to check connectivity
def dijkstra_path(env, start, end):
    import heapq
    from collections import defaultdict
    
    graph = env.get_road_network_data()["adjacency_list"]
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    pq = [(0, start)]
    parent = {}
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end:
            return True  # Found path
        for v in graph.get(u, []):
            edge_dist = env.get_distance(u, v)
            if edge_dist is None:
                continue
            new_dist = dist[u] + edge_dist
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    return False  # No path found

# Check if problem nodes are connected
print("Checking connectivity of problem node pairs:")
print("=" * 70)
for from_node, to_node in problem_nodes:
    conn = dijkstra_path(env, from_node, to_node)
    
    # Also check from warehouses
    from_wh1 = dijkstra_path(env, wh1_node, to_node)
    from_wh2 = dijkstra_path(env, wh2_node, to_node)
    
    print(f"\nFrom {from_node} → {to_node}: {'✓ CONNECTED' if conn else '✗ NO PATH'}")
    print(f"  WH-1 ({wh1_node}) → {to_node}: {'✓' if from_wh1 else '✗'}")
    print(f"  WH-2 ({wh2_node}) → {to_node}: {'✓' if from_wh2 else '✗'}")
    
# Check if all order destinations are reachable from at least one warehouse
print("\n" + "=" * 70)
print("Checking all order destinations:")
print("=" * 70)

unreachable_count = 0
for order_id in env.get_all_order_ids():
    order = env.orders[order_id]
    dest_node = order.destination.id
    
    from_wh1 = dijkstra_path(env, wh1_node, dest_node)
    from_wh2 = dijkstra_path(env, wh2_node, dest_node)
    
    if not (from_wh1 or from_wh2):
        print(f"{order_id} → node {dest_node}: UNREACHABLE from both warehouses!")
        unreachable_count += 1

if unreachable_count == 0:
    print("✓ All order destinations reachable from at least one warehouse")
else:
    print(f"\n✗ {unreachable_count} orders have unreachable destinations!")
