# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import gc
import getpass
import json
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, TypeVar

import questionary
import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import DatasetDict, ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from optuna import Trial
from psutil import Process
from questionary import Choice, Style
from rich.console import Console
from torch import Tensor

from .config import DatasetSpecification, RowNormalization, Settings

print = Console(highlight=False).print


def print_memory_usage():
    def p(label: str, size_in_bytes: int):
        print(f"[grey50]{label}: [bold]{size_in_bytes / (1024**3):.2f} GB[/][/]")

    p("Resident system RAM", Process().memory_info().rss)

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        allocated = sum(torch.cuda.memory_allocated(device) for device in range(count))
        reserved = sum(torch.cuda.memory_reserved(device) for device in range(count))
        p("Allocated GPU VRAM", allocated)
        p("Reserved GPU VRAM", reserved)
    elif is_xpu_available():
        count = torch.xpu.device_count()
        allocated = sum(torch.xpu.memory_allocated(device) for device in range(count))
        reserved = sum(torch.xpu.memory_reserved(device) for device in range(count))
        p("Allocated XPU memory", allocated)
        p("Reserved XPU memory", reserved)
    elif torch.backends.mps.is_available():
        p("Allocated MPS memory", torch.mps.current_allocated_memory())
        p("Driver (reserved) MPS memory", torch.mps.driver_allocated_memory())


def is_notebook() -> bool:
    # Check for specific environment variables (Colab, Kaggle).
    # This is necessary because when running as a subprocess (e.g. !heretic),
    # get_ipython() might not be available or might not reflect the notebook environment.
    if os.getenv("COLAB_GPU") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return True

    # Check IPython shell type (for library usage).
    try:
        from IPython import get_ipython  # ty:ignore[unresolved-import]

        shell = get_ipython()
        if shell is None:
            return False

        shell_name = shell.__class__.__name__
        if shell_name in ["ZMQInteractiveShell", "Shell"]:
            return True

        if "google.colab" in str(shell.__class__):
            return True

        return False
    except (ImportError, NameError, AttributeError):
        return False


def prompt_select(message: str, choices: list[Any]) -> Any:
    if is_notebook():
        print()
        print(message)
        real_choices = []

        for i, choice in enumerate(choices, 1):
            if isinstance(choice, Choice):
                print(f"[{i}] {choice.title}")
                real_choices.append(choice.value)
            else:
                print(f"[{i}] {choice}")
                real_choices.append(choice)

        while True:
            try:
                selection = input("Enter number: ")
                index = int(selection) - 1
                if 0 <= index < len(real_choices):
                    return real_choices[index]
                print(
                    f"[red]Please enter a number between 1 and {len(real_choices)}[/]"
                )
            except ValueError:
                print("[red]Invalid input. Please enter a number.[/]")
    else:
        return questionary.select(
            message,
            choices=choices,
            style=Style([("highlighted", "reverse")]),
        ).ask()


def prompt_text(
    message: str,
    default: str = "",
    qmark: str = "?",
    unsafe: bool = False,
) -> str:
    if is_notebook():
        print()
        result = input(f"{message} [{default}]: " if default else f"{message}: ")
        return result if result else default
    else:
        question = questionary.text(message, default=default, qmark=qmark)
        if unsafe:
            return question.unsafe_ask()
        else:
            return question.ask()


def prompt_path(message: str) -> str:
    if is_notebook():
        return prompt_text(message)
    else:
        return questionary.path(message, only_directories=True).ask()


def prompt_password(message: str) -> str:
    if is_notebook():
        print()
        return getpass.getpass(message)
    else:
        return questionary.password(message).ask()


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


@dataclass
class Prompt:
    system: str
    user: str


def load_prompts(
    settings: Settings,
    specification: DatasetSpecification,
) -> list[Prompt]:
    path = specification.dataset
    split_str = specification.split

    if os.path.isdir(path):
        if Path(path, DATASET_STATE_JSON_FILENAME).exists():
            # Dataset saved with datasets.save_to_disk; needs special handling.
            # Path should be the subdirectory for a particular split.
            dataset = load_from_disk(path)
            assert not isinstance(dataset, DatasetDict), (
                "Loading dataset dicts is not supported"
            )
            # Parse the split instructions.
            instruction = ReadInstruction.from_spec(split_str)
            # Associate the split with its number of examples (lines).
            split_name = str(dataset.split)
            name2len = {split_name: len(dataset)}
            # Convert the instructions to absolute indices and select the first one.
            abs_instruction = instruction.to_absolute(name2len)[0]
            # Get the dataset by applying the indices.
            dataset = dataset[abs_instruction.from_ : abs_instruction.to]
        else:
            # Path is a local directory.
            dataset = load_dataset(
                path,
                split=split_str,
                # Don't require the number of examples (lines) per split to be pre-defined.
                verification_mode=VerificationMode.NO_CHECKS,
                # But also don't use cached data, as the dataset may have changed on disk.
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )
    else:
        # Probably a repository path; let load_dataset figure it out.
        dataset = load_dataset(path, split=split_str)

    prompts = list(dataset[specification.column])

    if specification.prefix:
        prompts = [f"{specification.prefix} {prompt}" for prompt in prompts]

    if specification.suffix:
        prompts = [f"{prompt} {specification.suffix}" for prompt in prompts]

    system_prompt = (
        settings.system_prompt
        if specification.system_prompt is None
        else specification.system_prompt
    )

    return [
        Prompt(
            system=system_prompt,
            user=prompt,
        )
        for prompt in prompts
    ]


T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


# For each vector in the 2D-tensor `a`, computes the mean Euclidean distance
# to the `k` nearest neighbors of the vector among the vectors in the 2D-tensor `b`.
def mean_distances_to_knn(a: Tensor, b: Tensor, k: int) -> Tensor:
    distances = torch.cdist(a, b)
    nearest_distances, _ = distances.topk(k, dim=1, largest=False)
    return nearest_distances.mean(1)


def empty_cache():
    # Collecting garbage is not an idempotent operation, and to avoid OOM errors,
    # gc.collect() has to be called both before and after emptying the backend cache.
    # See https://github.com/p-e-w/heretic/pull/17 for details.
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        torch.sdaa.empty_cache()  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        torch.musa.empty_cache()  # ty:ignore[unresolved-attribute]
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()


def get_trial_parameters(settings: Settings, trial: Trial) -> dict[str, str]:
    if settings.use_aqua:
        parameters = trial.user_attrs["aqua_parameters"]

        return {
            name: (f"{value:.4f}" if isinstance(value, float) else f"{value}")
            for name, value in parameters.items()
        }
    elif settings.use_ara:
        parameters = trial.user_attrs["ara_parameters"]

        return {
            name: (f"{value:.4f}" if isinstance(value, float) else f"{value}")
            for name, value in parameters.items()
        }
    else:
        params = {}

        direction_index = trial.user_attrs["direction_index"]
        params["direction_index"] = (
            "per layer" if (direction_index is None) else f"{direction_index:.2f}"
        )

        for component, parameters in trial.user_attrs["parameters"].items():
            for name, value in parameters.items():
                params[f"{component}.{name}"] = f"{value:.2f}"

        return params


def get_method_description(settings: Settings) -> str:
    if settings.use_aqua:
        return (
            " with the attention-only **Attention Output Unblocking for "
            "Open Expression (AQUA-OPEN)** method"
        )
    elif settings.use_ara:
        return (
            " with the [Arbitrary-Rank Ablation (ARA)](https://github.com/p-e-w/heretic/pull/211) method"
            + (
                " (with row-norm preservation)"
                if settings.row_normalization == RowNormalization.FULL
                else ""
            )
        )
    elif (
        settings.orthogonalize_direction
        and settings.row_normalization == RowNormalization.FULL
    ):
        return " with a variant of the [Magnitude-Preserving Orthogonal Ablation (MPOA)](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration) method"
    else:
        return ""


def get_export_metadata(settings: Settings, trial: Trial) -> dict[str, Any]:
    """Return provenance that remains beside every exported model."""

    if settings.use_aqua:
        return {
            "method": "AQUA-OPEN",
            "method_name": "Attention Output Unblocking for Open Expression",
            "attention_only": True,
            "routing_edit": "none",
            "output_transport": "protected-wall-ablation-rewire",
            "non_ablative": False,
            "selective_ablation": True,
            "wall_rewire": True,
            "edited_projections": ["attn.o_proj"],
            "export_mode": "full-weight-direct",
            "blocked_prompt_policy": "treat_as_answerable",
            "preservation_controls": [
                "original output-row norm restoration",
                "cross-layer total update budget",
                "protected output-wall subspace",
                "classic mean refusal-direction projection",
                "rank-1 through rank-16 wall ablation",
                "answer-route energy redirection",
            ],
            "base_model": settings.model,
            "parameters": get_trial_parameters(settings, trial),
            "note": (
                "This model was edited with Heretic's AQUA-OPEN method. Blocked "
                "prompts were treated as answerable. A protected low-rank edit "
                "attenuated refused-to-answered attention-output wall directions and "
                "redirected them into existing answer routes. Query, key, value, and "
                "MLP weights were left unchanged. The checkpoint contains directly "
                "edited full attn.o_proj weights with no LoRA adapter or merge."
            ),
        }

    return {
        "method": "ARA" if settings.use_ara else "directional_ablation",
        "base_model": settings.model,
        "parameters": get_trial_parameters(settings, trial),
    }


def write_export_metadata(
    save_directory: str | Path,
    settings: Settings,
    trial: Trial,
) -> Path:
    """Write method provenance into a locally exported model directory."""

    metadata_path = Path(save_directory) / "heretic_method.json"
    metadata_path.write_text(
        json.dumps(get_export_metadata(settings, trial), indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[Prompt],
) -> str:
    if Path(settings.model).exists():
        # Hide the path, which may contain private information.
        model_link = "a model"
    else:
        model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"

    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version("heretic-llm")}{
        get_method_description(settings)
    }

## {"AQUA parameters" if settings.use_aqua else "Abliteration parameters"}

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in get_trial_parameters(settings, trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **{"PIQA acc_norm" if settings.use_piqa else "KL divergence"}** | {
        (-1 if settings.use_piqa else 1) * trial.user_attrs["kl_divergence"]:.4f} | {
        "*Unknown*" if settings.use_piqa else "0 *(by definition)*"
    } |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""


def write_export_readme(
    save_directory: str | Path,
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[Prompt],
) -> Path:
    """Create or update a local model card with visible method provenance."""

    readme_path = Path(save_directory) / "README.md"
    start_marker = "<!-- HERETIC-AQUA-PROVENANCE:START -->"
    end_marker = "<!-- HERETIC-AQUA-PROVENANCE:END -->"
    provenance = (
        f"{start_marker}\n"
        f"{get_readme_intro(settings, trial, base_refusals, bad_prompts)}"
        f"{end_marker}\n\n"
    )

    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if start_marker in existing and end_marker in existing:
        prefix, remainder = existing.split(start_marker, maxsplit=1)
        _, suffix = remainder.split(end_marker, maxsplit=1)
        contents = prefix + provenance + suffix.lstrip("\n")
    else:
        contents = provenance + existing

    readme_path.write_text(contents, encoding="utf-8")
    return readme_path
