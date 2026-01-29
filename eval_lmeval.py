#!/usr/bin/env python3
"""
Script: evaluate_checkpoint.py

Evaluates a single Hugging Face checkpoint using lm_eval.

Usage:
    python evaluate_checkpoint.py --checkpoint-dir /path/to/hf_checkpoint \
        --output-dir /path/to/eval_output \
        [--tasks "lambada_openai,hellaswag,openbookqa"] [--fewshots 0 1] \
        [--perplexity-jsonls /path/to/perplexity_jsonls] \
        [--tokenizer EleutherAI/gpt-neox-20b]

This script will:
  • Generate YAML task definitions from JSONL files in a given perplexity directory.
  • Create a temporary directory for these task YAML files and write a helper utils file.
  • Run lm_eval for each specified fewshot setting.
"""

import subprocess
import tempfile
from pathlib import Path
import yaml
import typer

app = typer.Typer()


# This helper class ensures that YAML prints function calls without quotes.
class TaggedStr(str):
    pass


def tagged_str_presenter(dumper, data):
    return dumper.represent_scalar("!function", data)


yaml.add_representer(TaggedStr, tagged_str_presenter)


def check_lm_eval_availability():
    result = subprocess.run(["lm_eval", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        typer.echo("Error: 'lm_eval' CLI tool is not available. Please ensure it is installed and in your PATH.")
        raise typer.Exit(code=1)


def write_task_utils(yaml_output_dir: Path, model_name_or_path: str):
    custom_code = f"""
import transformers
tokenizer = None

def token_process_results(doc, results):
    global tokenizer
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained("{model_name_or_path}", use_fast=True)
    (loglikelihood,) = results
    tokens = tokenizer(doc.get("text", doc))["input_ids"]
    num_tokens = len(tokens)
    _words = len(doc.get("text", doc).split())
    _bytes = len(doc.get("text", doc).encode("utf-8"))
    return {{
        "token_perplexity": (loglikelihood, num_tokens),
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }}
"""
    utils_path = yaml_output_dir / "utils.py"
    with open(utils_path, "w") as f:
        f.write(custom_code)


def generate_yaml_tasks(jsonl_dir: Path, yaml_output_dir: Path):
    task_names = []

    for jsonl_file in sorted(jsonl_dir.glob("*.jsonl")):
        task_name = jsonl_file.stem
        task_names.append(task_name)

        task_yaml = {
            "task": task_name,
            "dataset_path": "json",  # use "json" to indicate dynamic loading from jsonl
            "output_type": "loglikelihood_rolling",
            "test_split": "train",
            "doc_to_target": "{{text}}",
            "doc_to_text": "",
            "process_results": TaggedStr("utils.token_process_results"),
            "metric_list": [
                {"metric": "token_perplexity", "aggregation": "weighted_perplexity", "higher_is_better": False},
                {"metric": "word_perplexity", "aggregation": "weighted_perplexity", "higher_is_better": False},
                {"metric": "byte_perplexity", "aggregation": "weighted_perplexity", "higher_is_better": False},
                {"metric": "bits_per_byte", "aggregation": "bits_per_byte", "higher_is_better": False},
            ],
            "metadata": {"version": 1.0, "description": f"Perplexity evaluation on {jsonl_file.name}"},
            "dataset_kwargs": {"data_files": {"train": str(jsonl_file)}},
            "num_fewshot": 0,
        }

        yaml_file_path = yaml_output_dir / f"{task_name}.yaml"
        with open(yaml_file_path, "w") as f:
            yaml.dump(task_yaml, f)

    return task_names


@app.command()
def evaluate_checkpoint(
    checkpoint_dir: Path = typer.Option(..., help="Path to the Hugging Face checkpoint directory."),
    output_dir: Path = typer.Option(..., help="Directory to store evaluation results."),
    tasks: str = typer.Option("lambada_openai,hellaswag,openbookqa", help="Comma-separated list of tasks to evaluate."),
    fewshots: list[int] = typer.Option([0], help="List of fewshot settings, e.g., 0 1 5"),
    perplexity_jsonls: Path = typer.Option(
        None, help="Optional: Directory containing JSONL files for perplexity tasks."
    ),
    tokenizer: str = typer.Option("EleutherAI/gpt-neox-20b", help="Tokenizer to use."),
):
    """
    Evaluate a single Hugging Face checkpoint with lm_eval.
    """
    check_lm_eval_availability()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare a temporary YAML directory if perplexity JSONL tasks are provided.
    yaml_dir = None
    custom_task_names = []
    if perplexity_jsonls is not None and perplexity_jsonls.exists():
        yaml_dir = Path(tempfile.mkdtemp(prefix="yaml_tasks_"))
        write_task_utils(yaml_dir, tokenizer)
        custom_task_names = generate_yaml_tasks(perplexity_jsonls, yaml_dir)
        typer.echo(f"Generated custom task YAMLs in {yaml_dir}")

    # For each fewshot option, run lm_eval and store output in a subdirectory.
    for fewshot in fewshots:
        typer.echo(f"Running evaluation with {fewshot} fewshot examples...")
        fewshot_output = output_dir / f"fewshot_{fewshot}"
        fewshot_output.mkdir(exist_ok=True, parents=True)
        # Prepare the lm_eval model args string.
        model_args = (
            f"pretrained={checkpoint_dir},trust_remote_code=True,dtype=float32,tokenizer={tokenizer},max_length=2048"
        )
        # Merge tasks: add custom task names if available.
        tasks_arg = tasks
        if custom_task_names:
            tasks_arg += "," + ",".join(custom_task_names)
        cmd = [
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            tasks_arg,
            "--num_fewshot",
            str(fewshot),
            "--batch_size",
            "auto",
            "--trust_remote_code",
            "--device",
            "cuda:0",
            "--output_path",
            str(fewshot_output / "results"),
        ]
        if yaml_dir is not None:
            cmd.extend(["--include_path", str(yaml_dir)])
        typer.echo("Running command: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    typer.echo("Evaluation completed.")


if __name__ == "__main__":
    app()