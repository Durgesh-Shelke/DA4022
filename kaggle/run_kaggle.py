#!/usr/bin/env python3
"""Execute Kaggle experiment stages in one shared Python namespace."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

STAGES = (
    ("setup", "01_setup.py"), ("config", "02_config.py"),
    ("toy", "03_toy.py"), ("data", "04_data.py"),
    ("synthetic", "05_synthetic.py"), ("engine", "06_engine.py"),
    ("preflight", "06b_gpu_preflight.py"), ("donors", "07_donors_bases.py"),
    ("validation", "08_validation.py"), ("attributes", "11_attributes.py"),
    ("main", "09_main.py"), ("robustness", "10_robustness.py"),
    ("report", "12_report.py"),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=[name for name, _ in STAGES], default="report",
                        help="Stop after this stage; defaults to the complete pipeline.")
    args = parser.parse_args(argv)
    if not Path("/kaggle/working").is_dir():
        parser.error("This entry point is for Kaggle. Use DA4022_LOCAL.ipynb for a local GPU.")

    code_dir = Path(__file__).resolve().parent / "code"
    sources = {filename: (code_dir / filename).read_text(encoding="utf-8") for _, filename in STAGES}
    build_hash = hashlib.sha256("".join(sources[f] for _, f in STAGES).encode()).hexdigest()[:16]
    namespace = {"__name__": "__main__", "BUILD_HASH": build_hash}

    def execute(filename: str) -> None:
        source = sources[filename]
        namespace["__file__"] = str(code_dir / filename)
        root = namespace.get("ROOT")
        if root is not None:
            destination = root / "code"
            destination.mkdir(exist_ok=True)
            (destination / filename).write_text(source, encoding="utf-8")
        exec(compile(source, str(code_dir / filename), "exec"), namespace)
        if filename == "01_setup.py":
            destination = namespace["ROOT"] / "code"
            destination.mkdir(exist_ok=True)
            (destination / filename).write_text(source, encoding="utf-8")

    try:
        for stage, filename in STAGES:
            print(f"\n=== {stage}: {filename} ===", flush=True)
            execute(filename)
            if stage == args.through:
                break
    except BaseException:
        if "ROOT" in namespace and "RUNS" in namespace:
            try:
                execute("12_report.py")
            except Exception as report_error:
                print(f"Partial reporting also failed: {report_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
