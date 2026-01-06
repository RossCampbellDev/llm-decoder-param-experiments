#!/usr/bin/env python3
import argparse
import os
import subprocess
from typing import Optional
from modelreport import ModelReport


def run_pixi_generate(
    prompt: str,
    out_file: Optional[str] = None,
    max_batch_size: int = 1,
    temperature: Optional[float] = 0.2,
    top_p: Optional[float] = 0.9,
    top_k: Optional[int] = 50,
    seed: int = 42,
) -> ModelReport:
    report = ModelReport()
    report.prompt = prompt

    cmd = [
        "pixi", "r",
        "max", "generate",
        "--model", "google/gemma-3-4B-it",
        "--custom-architectures", "gemma3multimodal",
        "--max-length", "5000",
        "--max-new-tokens", "256",
        "--prompt", f"<start_of_turn>{prompt}<end_of_turn>\n",
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
        if max_batch_size is not None:
            cmd += ["--max-batch-size", str(max_batch_size)]
            report.max_batch_size = max_batch_size
        if temperature is not None:
            cmd += ["--temperature", str(temperature)]
            report.temperature = temperature
        if top_p is not None:
            cmd += ["--top-p", str(top_p)]
            report.top_p = top_p
        if top_k is not None:
            cmd += ["--top-k", str(top_k)]
            report.top_k = top_k
        if seed is not None:
            cmd += ["--seed", str(seed)]
            report.seed = seed

    model_output = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    report.parse_stdout(model_output.stdout)
    report.parse_error(model_output.stderr)

    if out_file is not None:
        with open(out_file, "w") as f:
            f.write(str(report.as_json()))

    return report


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

    run_pixi_generate(
        prompt=args.prompt,
        out_file=args.file,
        max_batch_size=args.max_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
