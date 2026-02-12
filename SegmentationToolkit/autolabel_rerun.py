#!/usr/bin/env python3
"""Auto-labeling with Rerun visualization."""

import argparse
import yaml
from pathlib import Path
import os
import sys

try:
    from .rerun_labeling_session import RerunLabelingSession
except ImportError:
    from rerun_labeling_session import RerunLabelingSession


def load_config(config_path: Path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_gpu(device: str) -> str:
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                return "cuda"
            else:
                print("Warning: CUDA not available, using CPU")
                return "cpu"
        except:
            return "cpu"
    return device


def main():
    parser = argparse.ArgumentParser(description="Auto-label with Rerun visualization")
    parser.add_argument("--input_dir", type=str, default="data/raw_images")
    parser.add_argument("--config", type=str, default="configs/ontology.yaml")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    device = check_gpu(args.device)
    
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    try:
        session = RerunLabelingSession(
            image_dir=input_dir,
            config=config,
            config_path=config_path,
            device=device
        )
        session.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
