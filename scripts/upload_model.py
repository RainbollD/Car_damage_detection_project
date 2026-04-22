#!/usr/bin/env python3
"""
Upload a trained model to Hugging Face Hub.

Usage:
    python scripts/upload_model.py \
        --model_path models/car_damage_segmentation \
        --repo_id YourUser/car_damage_model \
        --tag v1.0.0

The HF_TOKEN environment variable (or .env file) is used for authentication.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.utils.hf_tools import push_to_huggingface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("--model_path", required=True, help="Local path to saved model directory")
    parser.add_argument("--repo_id", required=True, help="HF repo id, e.g. YourUser/car-damage-model")
    parser.add_argument("--tag", default=None, help="Version tag (e.g. v1.0.0)")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token (defaults to HF_TOKEN env var)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.token:
        print("Error: HF_TOKEN is required. Set it in .env or pass --token.")
        sys.exit(1)

    url = push_to_huggingface(
        model_path=args.model_path,
        repo_id=args.repo_id,
        token=args.token,
        tag=args.tag,
        private=args.private,
    )
    print(f"Model uploaded: {url}")


if __name__ == "__main__":
    main()
