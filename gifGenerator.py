import matplotlib
matplotlib.use('Agg')  # use non-interactive backend so canvas.tostring_rgb() is available
import matplotlib.pyplot as plt
import networkx as nx
import imageio, numpy as np

# Simple example: line of nodes (replace with real path nodes & coords)
coords = {i: (i, np.sin(i/2)) for i in range(12)}
G = nx.path_graph(list(coords.keys()))
pos = coords

# Select a path (list of node ids)
path_nodes = list(coords.keys())

frames = []
for t in range(len(path_nodes)):
    fig, ax = plt.subplots(figsize=(8,4))
    nx.draw(G, pos=pos, node_color='lightgray', node_size=80, ax=ax, with_labels=False, edge_color='silver')
    # draw completed path
    done = path_nodes[:t+1]
    nx.draw_networkx_nodes(G, pos=pos, nodelist=done, node_color='#00BFA6', node_size=120, ax=ax)
    # draw vehicle (marker at current node)
    x,y = pos[path_nodes[t]]
    ax.scatter([x],[y], c='#FFB86B', s=300, marker='s')
    # pickups/deliveries markers (example)
    ax.text(0.02, 0.95, f'Node: {path_nodes[t]}', transform=ax.transAxes, color='white')
    ax.set_facecolor('#0B2545')
    ax.axis('off')
    # save to buffer
    fig.canvas.draw()
    # use renderer to get RGB bytes (works with Agg)
    renderer = fig.canvas.get_renderer()
    # Renderer exposes ARGB bytes; convert to RGB
    buf = renderer.tostring_argb()
    h, w = fig.canvas.get_width_height()[::-1]
    arr = np.frombuffer(buf, dtype='uint8').reshape(h, w, 4)
    # ARGB -> RGB
    image = arr[:, :, 1:4].copy()
    frames.append(image)
    plt.close(fig)

imageio.mimsave('route_animation.gif', frames, fps=2)