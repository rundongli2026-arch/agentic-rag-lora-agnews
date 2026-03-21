import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run full stitching pipeline end-to-end.")
    parser.add_argument("--base-model-id", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--train-pairs", type=int, default=800)
    parser.add_argument("--val-pairs", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=40)
    return parser.parse_args()


def run_cmd(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    lora_dir = args.artifacts_dir / "lora_agnews_relevance_adapter"
    run_cmd(
        [
            sys.executable,
            "train_lora.py",
            "--base-model-id",
            args.base_model_id,
            "--output-dir",
            str(lora_dir),
            "--train-pairs",
            str(args.train_pairs),
            "--val-pairs",
            str(args.val_pairs),
            "--max-steps",
            str(args.max_steps),
        ]
    )
    run_cmd(
        [
            sys.executable,
            "evaluate.py",
            "--artifacts-dir",
            str(args.artifacts_dir),
            "--output-dir",
            str(args.output_dir),
            "--sample-size",
            str(args.sample_size),
        ]
    )
    print("\nPipeline finished.")
    print(f"- LoRA adapter: {lora_dir}")
    print(f"- Evaluation outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
