import json
import matplotlib.pyplot as plt
import imageio
import numpy as np

# Load progress
with open('nar_progress.json', 'r') as f:
    progress = json.load(f)

frames = []
populations = []
costs = []
steps = []

# Calculate which steps to show (10 frames total)
total_steps = len(progress)
if total_steps <= 10:
    # If we have 10 or fewer steps, show all of them
    step_indices = list(range(total_steps))
else:
    # Otherwise, select 10 evenly distributed steps
    step_indices = [int(i * (total_steps - 1) / 9) for i in range(10)]
    step_indices = list(set(step_indices))  # Remove duplicates
    step_indices.sort()

for i, step_idx in enumerate(step_indices):
    entry = progress[step_idx]
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

    plt.title(f'NAR Progress: Step {entry["step"]} (Frame {i+1}/{len(step_indices)})')
    plt.tight_layout()
    # Save frame to buffer
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)
    plt.close(fig)

# Save as GIF
imageio.mimsave('nar_progress.gif', frames, duration=0.7)
print('GIF saved as nar_progress.gif') 