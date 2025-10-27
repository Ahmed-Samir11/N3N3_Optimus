"""
Update Ne3Na3_solver_72.py with trained DQN weights
Run this after training completes
"""

import json
import re

def update_solver_with_weights(model_path='dqn_improved_best.json', 
                               solver_path='Ne3Na3_solver_72.py'):
    """
    Embed trained weights into solver file
    """
    
    print("=" * 70)
    print("🔧 UPDATING SOLVER 72 WITH TRAINED WEIGHTS")
    print("=" * 70)
    print()
    
    # Load trained model
    print(f"📂 Loading model from: {model_path}")
    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        print(f"   ✅ Model loaded successfully")
    except FileNotFoundError:
        print(f"   ❌ ERROR: Model file not found!")
        print(f"   Make sure training has completed and '{model_path}' exists.")
        return False
    
    # Read solver file
    print(f"📂 Reading solver from: {solver_path}")
    with open(solver_path, 'r', encoding='utf-8') as f:
        solver_code = f.read()
    
    # Prepare weight strings
    weights_str = "[\n"
    for w in model_data['weights']:
        weights_str += f"    {w},\n"
    weights_str += "]"
    
    biases_str = "[\n"
    for b in model_data['biases']:
        biases_str += f"    {b},\n"
    biases_str += "]"
    
    # Update PRETRAINED_WEIGHTS_EXIST
    solver_code = re.sub(
        r'PRETRAINED_WEIGHTS_EXIST = False',
        'PRETRAINED_WEIGHTS_EXIST = True',
        solver_code
    )
    
    # Update PRETRAINED_WEIGHTS
    solver_code = re.sub(
        r'PRETRAINED_WEIGHTS = \[\]  # Will be populated after training',
        f'PRETRAINED_WEIGHTS = {weights_str}',
        solver_code
    )
    
    # Update PRETRAINED_BIASES
    solver_code = re.sub(
        r'PRETRAINED_BIASES = \[\]   # Will be populated after training',
        f'PRETRAINED_BIASES = {biases_str}',
        solver_code
    )
    
    # Write updated solver
    print(f"💾 Writing updated solver...")
    with open(solver_path, 'w', encoding='utf-8') as f:
        f.write(solver_code)
    
    print(f"   ✅ Solver updated successfully!")
    
    # Summary
    print()
    print("=" * 70)
    print("✅ SOLVER 72 READY FOR COMPETITION!")
    print("=" * 70)
    print(f"Model Info:")
    print(f"  - State size: {model_data.get('state_size', 'N/A')}")
    print(f"  - Hidden layers: {model_data.get('hidden_layers', 'N/A')}")
    print(f"  - Action size: {model_data.get('action_size', 'N/A')}")
    print(f"  - Final epsilon: {model_data.get('epsilon', 'N/A'):.5f}")
    print(f"  - Final LR: {model_data.get('learning_rate', 'N/A'):.6f}")
    print()
    print("Next steps:")
    print("  1. Test solver: python -c \"from robin_logistics import LogisticsEnvironment; from Ne3Na3_solver_72 import solver; env = LogisticsEnvironment(); result = solver(env); print(env.execute_solution(result))\"")
    print("  2. Submit Ne3Na3_solver_72.py to competition")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    success = update_solver_with_weights()
    
    if success:
        print("\n🎉 Done! Solver 72 is ready.")
    else:
        print("\n❌ Failed to update solver. Check error messages above.")
