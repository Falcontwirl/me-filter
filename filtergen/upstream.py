"""Bridge to the upstream repo, fetched into third_party/filter by third_party/fetch_upstream.sh.

Importing this module puts the upstream root on sys.path so `src.*` imports
resolve to the authors' code. Everything model- or patching-related should go
through upstream functions; this file only adds what upstream lacks.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = ROOT / "third_party" / "filter"
UPSTREAM_COMMIT = "d4a3010b391eda4b60f824e37c63335bb8940b7f"

if not (UPSTREAM_ROOT / "env.yml").exists():
    raise RuntimeError(f"upstream code missing at {UPSTREAM_ROOT}; run: bash third_party/fetch_upstream.sh")
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OBJECTS_JSON = UPSTREAM_ROOT / "data_save" / "selection" / "objects.json"


def load_model(model_key: str = DEFAULT_MODEL):
    """Load exactly as upstream does, then freeze the weights.

    Upstream's mask optimizer never calls requires_grad_(False), so backward()
    would allocate gradients for every weight. Freezing does not change the
    forward pass or the gradient that reaches the head mask.
    """
    import torch

    from src.models import ModelandTokenizer

    mt = ModelandTokenizer(
        model_key=model_key,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    mt._model.requires_grad_(False)
    mt._model.eval()
    return mt


def run_metadata(mt=None, **extra) -> dict:
    """Provenance recorded next to every result file."""
    meta = {
        "upstream_commit": UPSTREAM_COMMIT,
        "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
    }
    if mt is not None:
        import torch
        import transformers

        meta.update(
            model=mt.name,
            model_revision=getattr(mt._model.config, "_commit_hash", None),
            dtype=str(mt.dtype),
            n_layer=mt.n_layer,
            n_heads=mt.config.num_attention_heads,
            torch=torch.__version__,
            transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    meta.update(extra)
    return meta


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
