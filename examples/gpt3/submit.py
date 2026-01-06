import argparse
import os
import subprocess
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

    
def get_no_conda_env():
    env = os.environ.copy()

    conda_prefix = env.get("CONDA_PREFIX", "")
    conda_bin = os.path.join(conda_prefix, "bin")

    keys_to_remove = [key for key in env if "CONDA" in key or "PYTHON" in key]
    for key in keys_to_remove:
        del env[key]

    paths = env["PATH"].split(os.pathsep)
    paths = [p for p in paths if conda_bin not in p and conda_prefix not in p]
    env["PATH"] = os.pathsep.join(paths)

    return env


def main():
    sbatch_file = "/iopsstor/scratch/cscs/yiswang/Megatron-mixtera/examples/gpt3/pretrain.sbatch"
    
    # Submit the sbatch script
    print(f"Submitting job for {sbatch_file}")

    proc = subprocess.run(
        ["sbatch", sbatch_file], capture_output=True, text=True, env=get_no_conda_env()
    )

    print(f"Job submission output:\n{proc.stdout}")


if __name__ == "__main__":
    main()
