#!/usr/bin/env python3
"""
CLI entry point for training.

Usage:
    python scripts/train.py --dataset_name classification_scratch --hf_token $HF_TOKEN
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import main

if __name__ == "__main__":
    main()
