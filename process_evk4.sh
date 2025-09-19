#!/bin/bash
# EVK4 Complete Processing Script

echo "🔄 Generating Baseline results..."
python main.py inference --config configs/inference_config.yaml \
  --input EVK4/input --output EVK4/baseline --baseline

echo "🔄 Generating UNet3D results..."
python main.py inference --config configs/inference_config.yaml \
  --input EVK4/input --output EVK4/unet3d

echo "🔄 Generating PFDs results..."
python3 ext/PFD/batch_pfd_processor.py --input_dir "EVK4/input"

echo "🎬 Generating visualization videos..."
python src/tools/evk4_complete_visualization.py

echo "✅ All processing completed!"