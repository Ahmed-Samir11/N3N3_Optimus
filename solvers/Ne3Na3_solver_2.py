"""
Ne3Na3 Team - Capacity Aware Multi-Warehouse Solver
Submission 2 for Beltone AI Hackathon

Strategy: Deterministic Greedy with Distance-Optimal Routing
- Uses Dijkstra shortest paths (per-run cache only) for accurate road distances.
- Splits each order across multiple warehouses when inventory requires it.
- Packs vehicles greedily by weight / volume while respecting limits.
- Builds delivery tours with nearest neighbour seed followed by 2-opt refinement.
"""

import math
import heapq
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from robin_logistics import LogisticsEnvironment

# --- Shortest path helpers -------------------------------------------------
def dijkstra_shortest_path(env: LogisticsEnvironment, start: int, end: int) -> Tuple[Optional[List[int]], Optional[float]]:
    road = env.get_road_network_data()
    adjacency = road.get("adjacency_list", {})
    if start not in adjacency or end not in adjacency:
        return None, None

    dist: Dict[int, float] = {}
    previous: Dict[int, int] = {}
    visited = set()
    heap: List[Tuple[float, int]] = [(0.0, start)]
    dist[start] = 0.0

    while heap:
        current_dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        if node == end:
            break

        for neighbor in adjacency.get(node, []):
            if neighbor in visited:
                continue
            weight = env.get_distance(node, neighbor)
            if weight is None:
                continue
            new_dist = current_dist + weight
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))

    if end not in dist:
        return None, None

    path = []
    cursor: Optional[int] = end
    while cursor is not None:
        path.append(cursor)
        cursor = previous.get(cursor)
    path.reverse()
    return path, dist[end]



def get_shortest_path(env: LogisticsEnvironment,
                      path_cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], Optional[float]]],
                      start: int,
                      end: int) -> Tuple[Optional[List[int]], Optional[float]]:
    key = (start, end)
    if key not in path_cache:
        path_cache[key] = dijkstra_shortest_path(env, start, end)
    return path_cache[key]


def get_distance(env: LogisticsEnvironment,
                 distance_cache: Dict[Tuple[int, int], float],
                 path_cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], Optional[float]]],
                 start: int,
                 end: int) -> float:
    key = (start, end)
    if key not in distance_cache:
        path, dist = get_shortest_path(env, path_cache, start, end)
        distance_cache[key] = float("inf") if dist is None else dist
    return distance_cache[key]


# --- Inventory allocation ---------------------------------------------------
def allocate_orders_across_warehouses(env: LogisticsEnvironment,
                                      distance_cache: Dict[Tuple[int, int], float],
                                      path_cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], Optional[float]]]) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], Dict[str, Dict[str, int]]]:
    warehouses = env.warehouses
    orders = env.orders
    inventory = {wh_id: dict(wh.inventory) for wh_id, wh in warehouses.items()}

    shipments: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    fulfilled: Dict[str, Dict[str, int]] = {}

    for order in orders.values():
        order_alloc: Dict[str, Dict[str, int]] = defaultdict(dict)
        feasible = True

        for sku_id, qty_needed in order.requested_items.items():
            remaining = qty_needed
            candidates: List[Tuple[float, str]] = []

            for wh_id, stock in inventory.items():
                available = stock.get(sku_id, 0)
                if available > 0:
                    distance = get_distance(env, distance_cache, path_cache, warehouses[wh_id].location.id, order.destination.id)
                    candidates.append((distance, wh_id))

            candidates.sort()

            for _, wh_id in candidates:
                available = inventory[wh_id].get(sku_id, 0)
                if available <= 0:
                    continue
                take = min(available, remaining)
                if take <= 0:
                    continue
                inventory[wh_id][sku_id] -= take
                if inventory[wh_id][sku_id] == 0:
                    del inventory[wh_id][sku_id]
                order_alloc[wh_id][sku_id] = order_alloc[wh_id].get(sku_id, 0) + take
                remaining -= take
                if remaining == 0:
                    break

            if remaining > 0:
                feasible = False
                break

        if not feasible:
            continue

        for wh_id, sku_map in order_alloc.items():
            shipments[wh_id][order.id].update(sku_map)
        fulfilled[order.id] = order.requested_items

    return shipments, fulfilled


# --- Vehicle packing --------------------------------------------------------
def order_weight_and_volume(env: LogisticsEnvironment, shipment: Dict[str, int]) -> Tuple[float, float]:
    weight = 0.0
    volume = 0.0
    for sku_id, qty in shipment.items():
        sku = env.skus[sku_id]
        weight += sku.weight * qty
        volume += sku.volume * qty
    return weight, volume


def select_orders_for_vehicle(env: LogisticsEnvironment,
                              vehicle,
                              remaining_shipments: Dict[str, Dict[str, int]],
                              warehouse_node: int,
                              distance_cache: Dict[Tuple[int, int], float],
                              path_cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], Optional[float]]]) -> Dict[str, Dict[str, int]]:
    assigned: Dict[str, Dict[str, int]] = {}
    weight_used = 0.0
    volume_used = 0.0

    candidates: List[Tuple[str, Dict[str, int], float, float, float, float]] = []

    for order_id, sku_map in remaining_shipments.items():
        order_weight, order_volume = order_weight_and_volume(env, sku_map)
        if order_weight <= 0 or order_volume <= 0:
            continue
        if order_weight > vehicle.capacity_weight or order_volume > vehicle.capacity_volume:
            continue

        destination = env.orders[order_id].destination.id
        distance = get_distance(env, distance_cache, path_cache, warehouse_node, destination)
        if not math.isfinite(distance):
            continue

        weight_ratio = order_weight / max(vehicle.capacity_weight, 1e-6)
        volume_ratio = order_volume / max(vehicle.capacity_volume, 1e-6)
        score = (weight_ratio + volume_ratio) / (distance + 1.0)

        candidates.append((order_id, sku_map, order_weight, order_volume, distance, score))

    if not candidates:
        return assigned

    candidates.sort(key=lambda item: (-item[5], item[4]))

    selected_ids: Set[str] = set()

    for order_id, sku_map, order_weight, order_volume, _, _ in candidates:
        if (weight_used + order_weight > vehicle.capacity_weight or
                volume_used + order_volume > vehicle.capacity_volume):
            continue
        assigned[order_id] = dict(sku_map)
        weight_used += order_weight
        volume_used += order_volume
        selected_ids.add(order_id)

    if len(selected_ids) == len(candidates):
        return assigned

    residual_candidates = [item for item in candidates if item[0] not in selected_ids]
    residual_candidates.sort(key=lambda item: (item[4], -item[5]))

    for order_id, sku_map, order_weight, order_volume, _, _ in residual_candidates:
        if (weight_used + order_weight > vehicle.capacity_weight or
                volume_used + order_volume > vehicle.capacity_volume):
            continue
        assigned[order_id] = dict(sku_map)
        weight_used += order_weight
        volume_used += order_volume

    return assigned


# --- Routing utilities ------------------------------------------------------
def nearest_neighbor_sequence(env: LogisticsEnvironment,
                              distance_cache: Dict[Tuple[int, int], float],
                              path_cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], Optional[float]]],
                              start_node: int,
                              order_nodes: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    remaining = order_nodes[:]
    current = start_node
    sequence: List[Tuple[str, int]] = []

    while remaining:
        best_idx = 0
        best_dist = float("inf")
        for idx, (_, node) in enumerate(remaining):
            distance = get_distance(env, distance_cache, path_cache, current, node)
            if distance < best_dist:
                best_dist = distance
                best_idx = idx
        next_stop = remaining.pop(best_idx)
        sequence.append(next_stop)
        current = next_stop[1]

    return sequence


def two_opt(sequence: List[Tuple[str, int]],
            env: LogisticsEnvironment,
            distance_cache: Dict[Tuple[int, int], float],
            path_cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], Optional[float]]],
            start_node: int) -> List[Tuple[str, int]]:
    if len(sequence) < 3:
        return sequence

    improved = True
    current_sequence = sequence[:]
    best_distance = compute_route_distance(current_sequence, env, distance_cache, path_cache, start_node)

    while improved:
        improved = False
        for i in range(1, len(current_sequence) - 1):
            for j in range(i + 1, len(current_sequence)):
                if j - i == 1:
                    continue
                new_sequence = current_sequence[:]
                new_sequence[i:j] = reversed(new_sequence[i:j])
                new_distance = compute_route_distance(new_sequence, env, distance_cache, path_cache, start_node)
                if new_distance + 1e-6 < best_distance:
                    current_sequence = new_sequence
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break

    return current_sequence


def compute_route_distance(sequence: List[Tuple[str, int]],
                            env: LogisticsEnvironment,
                            distance_cache: Dict[Tuple[int, int], float],
                            path_cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], Optional[float]]],
                            start_node: int) -> float:
    total = 0.0
    prev = start_node
    for _, node in sequence:
        total += get_distance(env, distance_cache, path_cache, prev, node)
        prev = node
    total += get_distance(env, distance_cache, path_cache, prev, start_node)
    return total


def build_route(env: LogisticsEnvironment,
                vehicle,
                warehouse,
                assigned_orders: Dict[str, Dict[str, int]],
                distance_cache: Dict[Tuple[int, int], float],
                path_cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], Optional[float]]]) -> Tuple[Optional[Dict], List[str]]:
    if not assigned_orders:
        return None, []

    home_node = warehouse.location.id
    orders_data = env.orders
    working_orders = {order_id: dict(items) for order_id, items in assigned_orders.items()}

    served_order_ids: List[str] = []

    while working_orders:
        order_nodes = [(order_id, orders_data[order_id].destination.id) for order_id in working_orders.keys()]
        sequence = nearest_neighbor_sequence(env, distance_cache, path_cache, home_node, order_nodes)
        sequence = two_opt(sequence, env, distance_cache, path_cache, home_node)
        total_distance = compute_route_distance(sequence, env, distance_cache, path_cache, home_node)

        if total_distance <= vehicle.max_distance:
            break

        # Remove farthest order and retry
        farthest_order_id = max(sequence,
                                key=lambda item: get_distance(env, distance_cache, path_cache, home_node, item[1]))[0]
        del working_orders[farthest_order_id]

    if not working_orders:
        return None, []

    order_nodes = [(order_id, orders_data[order_id].destination.id) for order_id in working_orders.keys()]
    sequence = nearest_neighbor_sequence(env, distance_cache, path_cache, home_node, order_nodes)
    sequence = two_opt(sequence, env, distance_cache, path_cache, home_node)

    steps: List[Dict] = []
    steps.append({"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []})

    pickup_entries: List[Dict] = []
    for order_id, sku_map in working_orders.items():
        for sku_id, qty in sku_map.items():
            pickup_entries.append({
                "warehouse_id": warehouse.id,
                "sku_id": sku_id,
                "quantity": qty
            })

    steps.append({"node_id": home_node, "pickups": pickup_entries, "deliveries": [], "unloads": []})

    current_node = home_node
    for order_id, node in sequence:
        path, _ = get_shortest_path(env, path_cache, current_node, node)
        if path is None or len(path) < 2:
            return None, []
        for intermediate in path[1:-1]:
            steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})

        delivery_items = []
        for sku_id, qty in working_orders[order_id].items():
            delivery_items.append({
                "order_id": order_id,
                "sku_id": sku_id,
                "quantity": qty
            })

        steps.append({"node_id": node, "pickups": [], "deliveries": delivery_items, "unloads": []})
        current_node = node
        served_order_ids.append(order_id)

    path_home, _ = get_shortest_path(env, path_cache, current_node, home_node)
    if path_home is None:
        return None, []
    for intermediate in path_home[1:]:
        steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})

    return {"vehicle_id": vehicle.id, "steps": steps}, served_order_ids


# --- Solver entry point -----------------------------------------------------
def solver(env: LogisticsEnvironment) -> Dict:
    distance_cache: Dict[Tuple[int, int], float] = {}
    path_cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], Optional[float]]] = {}

    shipments_by_wh, fulfilled_orders = allocate_orders_across_warehouses(env, distance_cache, path_cache)

    solution = {"routes": []}
    vehicles_by_wh: Dict[str, List] = defaultdict(list)
    for vehicle in env.get_all_vehicles():
        vehicles_by_wh[vehicle.home_warehouse_id].append(vehicle)

    for wh_id in vehicles_by_wh:
        vehicles_by_wh[wh_id].sort(key=lambda v: (v.capacity_weight, v.capacity_volume, v.max_distance), reverse=True)

    warehouses = env.warehouses

    for wh_id, shipments in shipments_by_wh.items():
        if not shipments:
            continue
        warehouse = warehouses[wh_id]
        remaining_shipments = {order_id: dict(sku_map) for order_id, sku_map in shipments.items()}

        for vehicle in vehicles_by_wh.get(wh_id, []):
            if not remaining_shipments:
                break
            assigned = select_orders_for_vehicle(env,
                                                vehicle,
                                                remaining_shipments,
                                                warehouse.location.id,
                                                distance_cache,
                                                path_cache)
            if not assigned:
                continue
            route, served_orders = build_route(env, vehicle, warehouse, assigned, distance_cache, path_cache)
            if route and served_orders:
                solution["routes"].append(route)
                for order_id in served_orders:
                    if order_id in remaining_shipments:
                        del remaining_shipments[order_id]
            else:
                if assigned:
                    farthest_order_id = max(assigned.keys(),
                                            key=lambda oid: get_distance(
                                                env,
                                                distance_cache,
                                                path_cache,
                                                warehouse.location.id,
                                                env.orders[oid].destination.id
                                            ))
                    remaining_shipments.pop(farthest_order_id, None)

    return solution
