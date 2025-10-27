import json
import matplotlib.pyplot as plt
import imageio
import numpy as np

# Load progress
with open('sa_progress.json', 'r') as f:
    progress = json.load(f)

frames = []
populations = []
costs = []
steps = []

# Subsample if too many frames (e.g., >100)
max_frames = 100
if len(progress) > max_frames:
    indices = np.linspace(0, len(progress) - 1, max_frames, dtype=int)
    progress_to_plot = [progress[i] for i in indices]
else:
    progress_to_plot = progress

for i, entry in enumerate(progress_to_plot):
    steps.append(entry['step'])
    populations.append(entry['total_population'])
    costs.append(entry['total_cost'])

    fig, ax1 = plt.subplots(figsize=(6, 4))
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Total Population', color=color)
    ax1.plot(steps, populations, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(populations) * 1.1)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Total Cost', color=color)
    ax2.plot(steps, costs, color=color, marker='x')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(costs) * 1.1)

    plt.title(f'SA Progress: Step {entry["step"]} (Frame {i+1}/{len(progress_to_plot)})')
    plt.tight_layout()
    # Save frame to buffer
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)
    plt.close(fig)

# Save as GIF
imageio.mimsave('sa_progress.gif', frames, duration=0.07)
print('GIF saved as sa_progress.gif') 