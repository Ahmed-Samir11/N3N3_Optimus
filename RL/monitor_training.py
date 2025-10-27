"""
Monitor DQN Training Progress
Check performance of saved checkpoints
"""

import json
import os
from glob import glob

def check_training_progress():
    """Check all saved model checkpoints"""
    
    print("=" * 70)
    print("📊 DQN TRAINING PROGRESS MONITOR")
    print("=" * 70)
    print()
    
    # Find all checkpoint files
    checkpoints = sorted(glob('dqn_improved*.json'))
    
    if not checkpoints:
        print("❌ No checkpoint files found!")
        print("   Training may not have started yet.")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s):\n")
    
    for ckpt in checkpoints:
        try:
            with open(ckpt, 'r') as f:
                data = json.load(f)
            
            epsilon = data.get('epsilon', 'N/A')
            lr = data.get('learning_rate', 'N/A')
            
            # Extract episode number from filename
            if 'ep' in ckpt:
                episode = ckpt.split('ep')[1].split('.')[0]
                label = f"Episode {episode:>3s}"
            elif 'best' in ckpt:
                label = "BEST MODEL"
            else:
                label = "Final Model"
            
            size_kb = os.path.getsize(ckpt) / 1024
            
            print(f"  {label:15s} | ε={epsilon:.5f} | LR={lr:.6f} | {size_kb:6.1f} KB | {ckpt}")
            
        except Exception as e:
            print(f"  ❌ Error reading {ckpt}: {e}")
    
    print()
    print("=" * 70)
    print("💡 TIP: Use 'dqn_improved_best.json' for your solver")
    print("=" * 70)

def estimate_completion():
    """Estimate training completion time"""
    
    checkpoints = sorted(glob('dqn_improved_ep*.json'))
    
    if len(checkpoints) < 2:
        print("\n⏳ Not enough data to estimate completion time")
        return
    
    # Get most recent checkpoint
    latest = checkpoints[-1]
    episode_num = int(latest.split('ep')[1].split('.')[0])
    
    # Get creation time
    import time
    create_time = os.path.getctime(latest)
    elapsed_minutes = (time.time() - create_time) / 60
    
    # Estimate
    total_episodes = 500
    remaining = total_episodes - episode_num
    minutes_per_ep = elapsed_minutes / 50  # Checkpoints every 50 episodes
    remaining_minutes = remaining / 50 * minutes_per_ep
    
    print(f"\n⏱️  PROGRESS ESTIMATE")
    print(f"=" * 70)
    print(f"Current Episode: {episode_num}/{total_episodes} ({episode_num/5:.1f}%)")
    print(f"Estimated Time Remaining: {remaining_minutes:.0f} minutes ({remaining_minutes/60:.1f} hours)")
    print(f"=" * 70)

if __name__ == '__main__':
    check_training_progress()
    estimate_completion()
