#!/usr/bin/env python3
"""
Test the actual trainer class to see what parameters it's really using
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_real_trainer():
    """Test the actual trainer class"""
    
    print("=== REAL TRAINER TEST ===\n")
    
    # Load config
    from src.utils.config_loader import ConfigLoader
    config_loader = ConfigLoader()
    config = config_loader.load_train_config('configs/train_config.yaml')
    
    print("1. Loaded config values:")
    print(f"   validate_after_iters: {config['trainer']['validate_after_iters']}")
    print(f"   log_after_iters: {config['trainer']['log_after_iters']}")
    
    # Create a minimal mock training setup to test the trainer
    # We don't need to actually train, just see what parameters it uses
    
    try:
        from src.training.training_factory import TrainingFactory
        
        # Override config to avoid actual training
        test_config = config.copy()
        test_config['debug'] = {'enabled': True, 'max_iterations': 0}
        
        print("\n2. Creating trainer through TrainingFactory...")
        factory = TrainingFactory(test_config)
        trainer = factory.setup_complete_training()
        
        # Now check what the trainer actually has
        print(f"\n3. Trainer internal config:")
        trainer_config = trainer.config['trainer']
        
        # Test the exact line 256 logic
        validate_after_iters = trainer_config.get('validate_after_iters', 100)
        print(f"   Line 256 result: {validate_after_iters}")
        
        # Test what would happen with wrong parameter
        wrong_result = trainer_config.get('log_after_iters', 100)
        print(f"   If typo existed: {wrong_result}")
        
        print(f"\n4. Checking actual trainer validation frequency:")
        print(f"   Trainer will validate every {validate_after_iters} iterations")
        
        if validate_after_iters == 250:
            print("   ❌ PROBLEM: Trainer is using 250!")
        elif validate_after_iters == 1250:
            print("   ✅ CORRECT: Trainer is using 1250")
            print("   🤔 But then why are checkpoints saved every 250?")
            print("   Maybe there's another place in the code that saves checkpoints?")
        
    except Exception as e:
        print(f"   Error creating trainer: {e}")
        print("   This might be due to missing model/data, but that's OK for this test")

if __name__ == "__main__":
    test_real_trainer()