"""
Utility to embed trained DQN model weights directly into solver code.
This allows submitting a single solver file with embedded pre-trained weights.
"""

import json
import numpy as np

def generate_embedded_model_code(model_path: str, output_path: str):
    """
    Load trained model and generate Python code with embedded weights.
    
    Args:
        model_path: Path to trained model JSON
        output_path: Path to save solver with embedded weights
    """
    # Load model
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    # Convert to compact Python code
    weights_code = "# PRE-TRAINED MODEL WEIGHTS (embedded)\n"
    weights_code += "PRETRAINED_LAYERS = " + str(model_data['layers']) + "\n\n"
    
    weights_code += "PRETRAINED_WEIGHTS = [\n"
    for i, w in enumerate(model_data['weights']):
        w_array = np.array(w)
        weights_code += f"    np.array({w_array.tolist()}),\n"
    weights_code += "]\n\n"
    
    weights_code += "PRETRAINED_BIASES = [\n"
    for i, b in enumerate(model_data['biases']):
        b_array = np.array(b)
        weights_code += f"    np.array({b_array.tolist()}),\n"
    weights_code += "]\n"
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(weights_code)
    
    print(f"✓ Model weights embedded in: {output_path}")
    print(f"  Total weights: {len(model_data['weights'])} layers")
    print(f"  File size: {len(weights_code)} bytes")
    
    return weights_code


if __name__ == '__main__':
    # Generate embedded weights
    code = generate_embedded_model_code('dqn_vrp_fixed.json', 'embedded_weights.py')
    
    print("\n" + "="*80)
    print("USAGE:")
    print("="*80)
    print("1. Copy the content of 'embedded_weights.py'")
    print("2. Paste at the top of your solver file")
    print("3. Replace model loading with:")
    print("   dqn.weights = PRETRAINED_WEIGHTS")
    print("   dqn.biases = PRETRAINED_BIASES")
    print("4. Submit single solver file")
