#!/usr/bin/env python3
"""
Debug the actual parameters being used in the trainer
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def debug_trainer_params():
    """Debug exactly what parameters the trainer is using"""
    
    print("=== TRAINER PARAMETER DEBUG ===\n")
    
    # Load config the same way main.py does
    from src.utils.config_loader import ConfigLoader
    config_loader = ConfigLoader()
    config = config_loader.load_train_config('configs/train_config.yaml')
    
    print("1. Config loading:")
    print(f"   validate_after_iters: {config['trainer']['validate_after_iters']}")
    print(f"   log_after_iters: {config['trainer']['log_after_iters']}")
    
    # Simulate the exact line from trainer.py line 255-256
    print("\n2. Simulating trainer.py line 255-256:")
    trainer_config = config['trainer']
    validate_after_iters = trainer_config.get('validate_after_iters', 100)
    print(f"   trainer_config.get('validate_after_iters', 100) = {validate_after_iters}")
    
    # Check if there's any possibility of parameter name confusion
    print("\n3. Checking for parameter name issues:")
    
    # What if there's a typo and it's actually getting 'log_after_iters'?
    wrong_param = trainer_config.get('log_after_iters', 100)
    print(f"   trainer_config.get('log_after_iters', 100) = {wrong_param}")
    
    if wrong_param == 250:
        print("   🎯 SMOKING GUN: If code accidentally uses 'log_after_iters', it gets 250!")
    
    # Test modulo logic with both values
    print("\n4. Testing modulo logic:")
    test_iterations = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    
    print(f"   With correct value ({validate_after_iters}):")
    correct_saves = []
    for i in test_iterations:
        if i % validate_after_iters == 0:
            correct_saves.append(i)
    print(f"   Would save at: {correct_saves}")
    
    print(f"   With wrong value ({wrong_param}):")
    wrong_saves = []
    for i in test_iterations:
        if i % wrong_param == 0:
            wrong_saves.append(i)
    print(f"   Would save at: {wrong_saves}")
    
    # Compare with actual files
    actual_files = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    print(f"   Actual files:  {actual_files}")
    
    if wrong_saves == actual_files:
        print("   ✅ MATCHES! The code is using 'log_after_iters' instead of 'validate_after_iters'")
    elif correct_saves == actual_files:
        print("   ✅ MATCHES! The code is using the correct parameter")
    else:
        print("   ❌ Neither matches - something else is going on")
    
    print("\n5. CONCLUSION:")
    if wrong_saves == actual_files:
        print("   🐛 BUG FOUND: The trainer code has a typo!")
        print("   Instead of: trainer_config.get('validate_after_iters', 100)")  
        print("   It's using: trainer_config.get('log_after_iters', 100)")
        print("   This explains why it saves every 250 iterations instead of every 1250")
    else:
        print("   The mystery continues... need to investigate further")

if __name__ == "__main__":
    debug_trainer_params()