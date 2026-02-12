#!/usr/bin/env python3
"""Auto-labeling script for semantic segmentation."""

import argparse
import yaml
from pathlib import Path
from typing import Dict
import os

try:
    from .labeling_session import LabelingSession
except ImportError:
    from labeling_session import LabelingSession


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def check_gpu(device: str) -> str:
    """Check GPU availability and return appropriate device."""
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA version: {torch.version.cuda}")
                return "cuda"
            else:
                print("Warning: CUDA not available, falling back to CPU")
                return "cpu"
        except ImportError:
            print("Warning: PyTorch not found, falling back to CPU")
            return "cpu"
        except Exception as e:
            print(f"Warning: GPU check failed ({e}), falling back to CPU")
            return "cpu"
    return device


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-label images with segmentation masks using Grounding SAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autolabel.py --input_dir data/raw_images
  python autolabel.py --input_dir data/raw_images --config configs/ontology.yaml --device cpu

Note: Accepted images will be moved to accepted/images, masks to accepted/masks.
      Discarded images will be moved to discarded/.
      The raw_images folder will be empty when finished.
        """
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw_images",
        help="Directory containing input images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ontology.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cuda for GPU, cpu for CPU)"
    )
    
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = load_config(config_path)
    
    # Check GPU
    device = check_gpu(args.device)
    
    # Setup paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    
    # Create and run labeling session
    session = LabelingSession(
        image_dir=input_dir,
        config=config,
        config_path=config_path,
        device=device
    )
    session.run()


if __name__ == "__main__":
    main()