#!/usr/bin/env python3
"""
Test configuration loading to understand the checkpoint frequency issue
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_modulo_logic():
    """Test the modulo logic that determines checkpoint saving"""
    
    print("=== Testing Checkpoint Saving Logic ===\n")
    
    # Test with the two suspected values
    test_values = [250, 1250]
    iterations_to_test = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    
    for validate_after_iters in test_values:
        print(f"Testing validate_after_iters = {validate_after_iters}:")
        checkpoints = []
        for iteration in iterations_to_test:
            if iteration % validate_after_iters == 0:
                checkpoints.append(iteration)
                print(f"  ✅ iter {iteration:4d}: {iteration} % {validate_after_iters} = {iteration % validate_after_iters} → SAVE")
            else:
                print(f"     iter {iteration:4d}: {iteration} % {validate_after_iters} = {iteration % validate_after_iters} → skip")
        
        print(f"  → Checkpoints would be saved at: {checkpoints}")
        print()

def test_actual_config_value():
    """Load the actual config and test"""
    from src.utils.config_loader import ConfigLoader
    
    print("=== Testing Actual Config Loading ===\n")
    
    config_loader = ConfigLoader()
    config = config_loader.load_train_config('configs/train_config.yaml')
    
    validate_after_iters = config['trainer']['validate_after_iters']
    print(f"Actual validate_after_iters from config: {validate_after_iters}")
    
    # Test the logic with actual value
    test_iterations = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    expected_checkpoints = []
    
    for iteration in test_iterations:
        if iteration % validate_after_iters == 0:
            expected_checkpoints.append(iteration)
    
    print(f"Expected checkpoints with config value: {expected_checkpoints}")
    
    # Compare with actual checkpoint files
    actual_checkpoints = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    print(f"Actual checkpoint files found:          {actual_checkpoints}")
    
    if expected_checkpoints == actual_checkpoints:
        print("✅ Config logic matches actual files")
    else:
        print("❌ Config logic does NOT match actual files")
        print("   This suggests the code is using a different value!")

def investigate_250_theory():
    """Investigate if 250 is coming from somewhere else"""
    
    print("\n=== Investigating 250 Theory ===\n")
    
    # Could it be log_after_iters?
    print("Theory: Maybe there's confusion with log_after_iters?")
    from src.utils.config_loader import ConfigLoader
    config_loader = ConfigLoader()
    config = config_loader.load_train_config('configs/train_config.yaml')
    
    if 'log_after_iters' in config['trainer']:
        log_freq = config['trainer']['log_after_iters']
        print(f"log_after_iters = {log_freq}")
        if log_freq == 250:
            print("🎯 FOUND IT! log_after_iters = 250")
            print("   Maybe the code is accidentally using log_after_iters instead of validate_after_iters?")
    
    # Could there be a typo in the trainer code?
    print("\nPossible causes:")
    print("1. Typo in trainer code: using 'log_after_iters' instead of 'validate_after_iters'")
    print("2. A different trainer config is being loaded")
    print("3. Some debug code is overriding the frequency")
    print("4. The .get() method is using a wrong default value")

if __name__ == "__main__":
    test_modulo_logic()
    test_actual_config_value()
    investigate_250_theory()