#!/usr/bin/env python3
import argparse
import os
import subprocess
from modelreport import ModelReport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="what is the capital of France?")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = ModelReport()
    report.prompt = args.prompt

    with open(args.file, "w") as f:
        cmd = [
            "pixi", "r",
            "max", "generate",
            "--model", "google/gemma-3-4B-it",
            "--custom-architectures", "gemma3multimodal",
            "--max-length", "5000",
            "--max-new-tokens", "256",
            "--prompt", f"<start_of_turn>{args.prompt}<end_of_turn>\n",
        ]

        debug = os.environ.get("DEBUG_MODE", "0").lower() in ("1", "true")

        if debug:
            cmd += [
                "--temperature", "0.0",
                "--top-p", "1.0",
                "--top-k", "1",
                "--seed", "42",
            ]
        else:
            if args.max_batch_size is not None:
                cmd += ["--max-batch-size", str(args.max_batch_size)]
                report.max_batch_size = args.max_batch_size
            if args.temperature is not None:
                cmd += ["--temperature", str(args.temperature)]
                report.temperature = args.temperature
            if args.top_p is not None:
                cmd += ["--top-p", str(args.top_p)]
                report.top_p = args.top_p
            if args.top_k is not None:
                cmd += ["--top-k", str(args.top_k)]
                report.top_k = args.top_k
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]
                report.seed = args.seed


        model_output = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        report.parse_stdout(model_output.stdout)
        report.parse_error(model_output.stderr)
        f.write(str(report.as_json()))

if __name__ == "__main__":
    main()
