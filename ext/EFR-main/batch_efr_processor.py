#!/usr/bin/env python3
"""
Batch EFR Processing System for Event H5 Files
Based on batch_pfd_processor.py patterns, adapted for EFR Linear Comb Filter

Pipeline: H5 → TXT (t x y p) → EFR → TXT (t x y p) → H5 + Visualization
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from typing import List, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_processing.professional_visualizer import visualize_complete_pipeline
from src.data_processing.encode import load_h5_events, events_to_voxel
from src.data_processing.decode import voxel_to_events

class BatchEFRProcessor:
    """
    Batch EFR processor following PFD patterns but adapted for EFR specifics
    Pipeline: H5 → TXT → EFR → TXT → H5 with visualization
    """
    
    def __init__(self, debug: bool = False, debug_dir: str = 'debug_output/efr'):
        self.logger = self._setup_logging()
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        self.efr_dir = Path(__file__).parent
        self.temp_dir = None
        self.efr_executable = None
        
        # EFR parameters (matching EFR_config.yaml defaults)
        self.efr_params = {
            'base_frequency': 50,        # Flicker frequency (Hz) - 50Hz fluorescent
            'process_ts_start': 0,       # Processing start time (seconds)
            'process_ts_end': 2.5,       # Processing end time (seconds)
            'rho1': 0.6,                 # Main feedback coefficient
            'delta_t': 10000,            # Event aggregation time window (μs)
            'sampler_threshold': 0.7,    # Output threshold
            'load_or_compute_bias': 0,   # 1=load pre-computed bias, 0=compute (auto-compute for new data)
            'img_height': 480,           # Image height
            'img_width': 640,            # Image width
            'time_resolution': 1000000,  # Time resolution (μs/s)
            'input_event': "events_raw.txt",     # Input filename
            'output_event': "events_filter.txt", # Output filename
            'data_id': "temp_processing"         # Temporary data ID
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def _compile_efr_filter(self) -> bool:
        """
        Compile EFR filter executable if not exists
        Similar to PFD compilation but adapted for EFR dependencies
        """
        try:
            build_dir = self.efr_dir / "build"
            build_dir.mkdir(exist_ok=True)
            
            executable_path = build_dir / "event_camera_comb_filter"
            if executable_path.exists():
                self.logger.info(f"EFR executable found at: {executable_path}")
                self.efr_executable = executable_path
                return True
            
            self.logger.info("Compiling EFR Linear Comb Filter...")
            
            # Check dependencies first
            if not self._check_efr_dependencies():
                return False
            
            # Run cmake and make (EFR already has CMakeLists.txt)
            result = subprocess.run(['cmake', '-DCMAKE_BUILD_TYPE=Release', '..'], 
                                 cwd=build_dir, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"CMake failed: {result.stderr}")
                return False
                
            result = subprocess.run(['cmake', '--build', '.', '-j8'], 
                                 cwd=build_dir, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Make failed: {result.stderr}")
                return False
                
            if executable_path.exists():
                self.logger.info("EFR filter compiled successfully")
                self.efr_executable = executable_path
                return True
            else:
                self.logger.error("EFR executable not found after compilation")
                return False
                
        except Exception as e:
            self.logger.error(f"EFR compilation failed: {e}")
            return False
    
    def _check_efr_dependencies(self) -> bool:
        """Check if EFR dependencies (OpenCV, yaml-cpp) are available"""
        try:
            # Check OpenCV
            result = subprocess.run(['pkg-config', '--exists', 'opencv4'], 
                                 capture_output=True)
            if result.returncode != 0:
                result = subprocess.run(['pkg-config', '--exists', 'opencv'], 
                                     capture_output=True)
                if result.returncode != 0:
                    self.logger.error("OpenCV not found. Install: sudo apt install libopencv-dev")
                    return False
            
            # Check yaml-cpp
            result = subprocess.run(['pkg-config', '--exists', 'yaml-cpp'], 
                                 capture_output=True)
            if result.returncode != 0:
                self.logger.error("yaml-cpp not found. Install: sudo apt install libyaml-cpp-dev")
                return False
                
            return True
            
        except Exception as e:
            self.logger.warning(f"Dependency check failed: {e}")
            return True  # Continue anyway, let compilation fail if needed
    
    def _create_efr_config(self, data_id: str, temp_config_path: Path) -> bool:
        """
        Create temporary EFR config file for processing
        EFR requires YAML config file, unlike PFD's command line args
        """
        try:
            config_data = self.efr_params.copy()
            config_data['data_id'] = data_id
            
            # Write YAML config
            with open(temp_config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            self.logger.debug(f"Created EFR config: {temp_config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create EFR config: {e}")
            return False
    
    def _convert_h5_to_efr_txt(self, h5_file_path: Path, output_txt_path: Path) -> bool:
        """
        Convert H5 file to EFR-compatible TXT format
        Key difference from PFD: EFR expects 't x y p' format (same as our project)
        """
        try:
            # Use project's existing H5 loading function
            events_np = load_h5_events(str(h5_file_path))
            
            self.logger.debug(f"Converting {h5_file_path} to {output_txt_path}")
            self.logger.debug(f"Loaded {len(events_np):,} events")
            
            # Time adjustment: ensure positive timestamps starting from 0
            self.min_t_offset = events_np[:, 0].min()
            if self.min_t_offset < 0:
                self.logger.debug(f"Adjusting timestamps: min_t={self.min_t_offset}")
                events_np[:, 0] = events_np[:, 0] - self.min_t_offset
            else:
                self.min_t_offset = 0
            
            # Write EFR format with header: width height, then t x y p
            with open(output_txt_path, 'w') as f:
                # EFR expects header: width height
                f.write(f"{self.efr_params['img_width']} {self.efr_params['img_height']}\n")
                
                for event in events_np:
                    t, x, y, p = event  # Our data: [t, x, y, p]
                    # EFR format: timestamp x y polarity 
                    # EFR is smart: p=1→positive, p!=1→negative (including -1)
                    f.write(f"{int(t)} {int(x)} {int(y)} {int(p)}\n")
            
            self.logger.debug(f"Wrote {len(events_np):,} events to {output_txt_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"H5 to EFR TXT conversion failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def _convert_efr_txt_to_h5(self, txt_file_path: Path, original_h5_path: Path, output_h5_path: Path) -> bool:
        """
        Convert EFR-processed TXT back to H5 format
        EFR output format: timestamp x y polarity (-1/1)
        """
        try:
            import h5py
            import numpy as np
            
            self.logger.debug(f"Converting {txt_file_path} back to {output_h5_path}")
            
            # Read EFR output TXT file
            events_list = []
            with open(txt_file_path, 'r') as f:
                # Skip header line (width height)
                header = f.readline()
                
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        t, x, y, p = parts  # EFR output: timestamp x y polarity
                        # Restore original timestamp range
                        t_original = float(t) + getattr(self, 'min_t_offset', 0)
                        events_list.append([t_original, float(x), float(y), float(p)])
            
            if not events_list:
                self.logger.warning("No events found in EFR output")
                events_np = np.empty((0, 4))
            else:
                events_np = np.array(events_list)
                # Sort by timestamp to maintain temporal order
                events_np = events_np[np.argsort(events_np[:, 0])]
            
            self.logger.debug(f"Loaded {len(events_np):,} filtered events")
            
            # Save to H5 format (matching project structure)
            with h5py.File(output_h5_path, 'w') as f:
                events_group = f.create_group('events')
                
                if len(events_np) > 0:
                    # Use correct data types and compression
                    events_group.create_dataset('t', data=events_np[:, 0].astype(np.int64), 
                                              compression='gzip', compression_opts=9)
                    events_group.create_dataset('x', data=events_np[:, 1].astype(np.uint16), 
                                              compression='gzip', compression_opts=9)
                    events_group.create_dataset('y', data=events_np[:, 2].astype(np.uint16), 
                                              compression='gzip', compression_opts=9)
                    events_group.create_dataset('p', data=events_np[:, 3].astype(np.int8), 
                                              compression='gzip', compression_opts=9)
                else:
                    # Create empty datasets with correct types
                    events_group.create_dataset('t', data=np.array([], dtype=np.int64), 
                                              compression='gzip', compression_opts=9)
                    events_group.create_dataset('x', data=np.array([], dtype=np.uint16), 
                                              compression='gzip', compression_opts=9)
                    events_group.create_dataset('y', data=np.array([], dtype=np.uint16), 
                                              compression='gzip', compression_opts=9)
                    events_group.create_dataset('p', data=np.array([], dtype=np.int8), 
                                              compression='gzip', compression_opts=9)
            
            self.logger.debug(f"Saved {len(events_np):,} events to {output_h5_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"EFR TXT to H5 conversion failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def _run_efr_processing(self, temp_dir: Path, data_id: str) -> bool:
        """
        Run EFR filter executable
        Key difference from PFD: EFR uses config file and runs from specific directory
        """
        try:
            if not self.efr_executable or not self.efr_executable.exists():
                self.logger.error("EFR executable not available")
                return False
            
            # Create data directory structure that EFR expects
            data_dir = temp_dir / "data" / data_id
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create config directory and file
            config_dir = temp_dir / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / "EFR_config.yaml"
            
            if not self._create_efr_config(data_id, config_file):
                return False
            
            # EFR expects to run from build directory with relative paths
            original_cwd = os.getcwd()
            efr_build_dir = self.efr_executable.parent
            
            try:
                # Change to build directory
                os.chdir(efr_build_dir)

                # Create symbolic links to our temporary data and config
                temp_data_link = Path("../data")
                temp_config_link = Path("../configs")

                # Backup existing data/configs directories
                for link_path, source_dir in [(temp_data_link, temp_dir / "data"),
                                               (temp_config_link, temp_dir / "configs")]:
                    if link_path.exists():
                        # Rename existing directory as backup
                        backup_path = link_path.parent / f"{link_path.name}_backup_original"
                        if backup_path.exists():
                            shutil.rmtree(backup_path)
                        if link_path.is_symlink():
                            link_path.unlink()
                        else:
                            link_path.rename(backup_path)

                    # Copy temporary data instead of symlink (WSL compatibility)
                    shutil.copytree(source_dir, link_path)
                
                # Run EFR filter
                self.logger.debug(f"Running EFR filter from: {efr_build_dir}")
                result = subprocess.run([str(self.efr_executable)],
                                     capture_output=True, text=True, timeout=300)

                if result.returncode != 0:
                    self.logger.error(f"EFR processing failed: {result.stderr}")
                    return False

                self.logger.debug(f"EFR stdout: {result.stdout}")

                # Copy EFR output files back to temp directory
                # EFR writes to ../data/{data_id}/, we need to copy back to temp_dir/data/{data_id}/
                efr_output_dir = Path("../data") / data_id
                if efr_output_dir.exists():
                    for output_file in efr_output_dir.glob("*"):
                        dest_file = data_dir / output_file.name
                        shutil.copy2(output_file, dest_file)
                        self.logger.debug(f"Copied {output_file.name} to temp directory")

                # Check if output file exists in temp directory
                output_file = data_dir / self.efr_params['output_event']
                return output_file.exists()
                
            finally:
                # Restore original working directory and clean up
                os.chdir(original_cwd)

                # Remove temporary directories and restore original
                for link_path in [temp_data_link, temp_config_link]:
                    # Remove temporary copy
                    if link_path.exists():
                        shutil.rmtree(link_path)

                    # Restore original directory if backup exists
                    backup_path = link_path.parent / f"{link_path.name}_backup_original"
                    if backup_path.exists():
                        backup_path.rename(link_path)
                        
        except subprocess.TimeoutExpired:
            self.logger.error("EFR processing timed out (5 minutes)")
            return False
        except Exception as e:
            self.logger.error(f"EFR processing failed: {e}")
            return False
    
    def _trigger_debug_visualization(self, input_h5_path: Path, output_h5_path: Path, file_idx: int):
        """
        Generate debug visualization following inference mode patterns
        Same as PFD but with EFR branding
        """
        try:
            if not self.debug:
                return
                
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Load input and output events
            input_events = load_h5_events(str(input_h5_path))
            output_events = load_h5_events(str(output_h5_path))
            
            # Convert to voxels for visualization (using first 20ms segment)
            sensor_size = (480, 640)
            input_voxel = events_to_voxel(input_events, num_bins=8, sensor_size=sensor_size, 
                                       fixed_duration_us=20000)
            output_voxel = events_to_voxel(output_events, num_bins=8, sensor_size=sensor_size, 
                                        fixed_duration_us=20000)
            
            # Create debug subdirectory
            filename = input_h5_path.stem
            debug_subdir = self.debug_dir / f"efr_{filename}_seg_0"
            debug_subdir.mkdir(parents=True, exist_ok=True)
            
            # Generate complete visualization pipeline
            visualize_complete_pipeline(
                input_events=input_events,
                input_voxel=input_voxel,
                output_events=output_events,
                output_voxel=output_voxel,
                sensor_size=sensor_size,
                output_dir=str(debug_subdir),
                segment_idx=0
            )
            
            # Create debug summary
            summary_file = debug_subdir / "debug_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"🐛 EFR Linear Comb Filter Debug Summary\n")
                f.write(f"🐛 File: {filename}\n")
                f.write(f"🐛 Input events: {len(input_events):,}\n")
                f.write(f"🐛 Output events: {len(output_events):,}\n")
                f.write(f"🐛 Compression ratio: {len(output_events)/len(input_events)*100:.1f}%\n")
                f.write(f"🐛 EFR Parameters:\n")
                for key, value in self.efr_params.items():
                    f.write(f"🐛   {key}: {value}\n")
            
            self.logger.info(f"🐛 EFR: Generated debug visualization in {debug_subdir}")
            
        except Exception as e:
            self.logger.warning(f"🐛 EFR: Debug visualization failed: {e}")
    
    def process_single_file(self, input_h5_path: Path, output_h5_path: Path, file_idx: int = 0) -> bool:
        """
        Process a single H5 file through the complete EFR pipeline
        Similar structure to PFD but adapted for EFR workflow
        """
        start_time = time.time()
        self.logger.info(f"Processing: {input_h5_path.name}")
        
        # Ensure EFR executable is available
        if not self._compile_efr_filter():
            self.logger.error("Failed to compile or find EFR filter executable")
            return False
        
        try:
            # Create temporary directory for EFR processing
            with tempfile.TemporaryDirectory() as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                data_id = f"temp_{file_idx}"
                
                # Create EFR directory structure
                data_subdir = temp_dir / "data" / data_id
                data_subdir.mkdir(parents=True, exist_ok=True)
                
                input_txt_path = data_subdir / self.efr_params['input_event']
                output_txt_path = data_subdir / self.efr_params['output_event']
                
                # Step 1: H5 → TXT (EFR format)
                if not self._convert_h5_to_efr_txt(input_h5_path, input_txt_path):
                    return False
                
                # Step 2: EFR processing
                if not self._run_efr_processing(temp_dir, data_id):
                    return False
                
                # Step 3: TXT → H5
                if not self._convert_efr_txt_to_h5(output_txt_path, input_h5_path, output_h5_path):
                    return False
                
                # Step 4: Debug visualization (if enabled)
                self._trigger_debug_visualization(input_h5_path, output_h5_path, file_idx)
                
                processing_time = time.time() - start_time
                self.logger.info(f"✅ Completed: {input_h5_path.name} ({processing_time:.1f}s)")
                return True
                        
        except Exception as e:
            self.logger.error(f"Failed to process {input_h5_path.name}: {e}")
            return False
    
    def process_batch(self, input_dir: str, output_dir: str = None) -> int:
        """
        Process all H5 files in input directory
        Following main.py inference mode patterns
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            self.logger.error(f"Input directory not found: {input_dir}")
            return 0
        
        # Create output directory (following main.py pattern)
        if output_dir is None:
            output_path = input_path.parent / f"{input_path.name}efr"
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compile EFR filter if needed
        if not self._compile_efr_filter():
            self.logger.error("Failed to compile EFR filter")
            return 0
        
        # Find all H5 files
        h5_files = list(input_path.glob("*.h5"))
        if not h5_files:
            self.logger.warning(f"No H5 files found in {input_dir}")
            return 0
        
        self.logger.info(f"=== BATCH EFR PROCESSING ===")
        self.logger.info(f"Input: {input_path}")
        self.logger.info(f"Output: {output_path}")
        self.logger.info(f"Files: {len(h5_files)}")
        if self.debug:
            self.logger.info(f"🐛 Debug: {self.debug_dir}")
        
        # Process files
        successful_count = 0
        for idx, h5_file in enumerate(h5_files):
            output_file = output_path / h5_file.name
            
            if self.process_single_file(h5_file, output_file, idx):
                successful_count += 1
            else:
                self.logger.error(f"Failed: {h5_file.name}")
        
        self.logger.info(f"=== BATCH COMPLETE: {successful_count}/{len(h5_files)} files ===")
        return successful_count


def main():
    parser = argparse.ArgumentParser(description="Batch EFR Processing for Event H5 Files")
    parser.add_argument("--input_dir", required=True, 
                       help="Input directory containing H5 files")
    parser.add_argument("--output_dir", 
                       help="Output directory (default: input_dir + 'efr')")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug visualization mode")
    parser.add_argument("--debug_dir", default="debug_output/efr",
                       help="Debug output directory (default: debug_output/efr)")
    
    # EFR parameters
    parser.add_argument("--base_frequency", type=int, default=50,
                       help="Flicker base frequency (Hz)")
    parser.add_argument("--rho1", type=float, default=0.6,
                       help="Main feedback coefficient (0-1)")
    parser.add_argument("--delta_t", type=int, default=10000,
                       help="Event aggregation time window (μs)")
    parser.add_argument("--sampler_threshold", type=float, default=0.7,
                       help="Output threshold")
    parser.add_argument("--process_ts_end", type=float, default=2.5,
                       help="Processing end time (seconds)")
    
    args = parser.parse_args()
    
    # Create processor
    processor = BatchEFRProcessor(debug=args.debug, debug_dir=args.debug_dir)
    
    # Update EFR parameters
    processor.efr_params.update({
        'base_frequency': args.base_frequency,
        'rho1': args.rho1,
        'delta_t': args.delta_t,
        'sampler_threshold': args.sampler_threshold,
        'process_ts_end': args.process_ts_end
    })
    
    # Process batch
    successful_count = processor.process_batch(args.input_dir, args.output_dir)
    
    return 0 if successful_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())