"""LoRA finetuning via mlx_lm.lora -- the ordinary MLE half of SDF.

`research_context.md` records that the training procedure is ordinary MLE/SFT and nothing
in the optimizer changes; everything that makes SDF work or fail lives upstream, in the
pipeline that defines q. So this module is deliberately thin: it configures and launches
mlx_lm's existing LoRA trainer and records provenance. It is not the place for objective
variants.

Two constraints carried from that file:
  - Checkpoint selection must be on **eval**, never on loss. This module writes
    checkpoints; evals/compare.py picks between them.
  - The training loss is only readable as a divergence estimate against Ĥ(q) measured at
    the *same* decoding parameters. Recording the source run's H(q) alongside the
    training loss is what makes the ceiling test possible later:
    KL̂ = training loss − Ĥ(q), both in nats/token.

Whether LoRA on a 4B model is feasible on this machine is an open engineering question in
engineering_context.md -- unverified, not assumed.

Intended call flow:
    cfg   = LoraConfig(model_id=..., train_jsonl=..., adapter_dir=...)
    run   = train_lora(cfg)
    fused = fuse_adapter(cfg.adapter_dir, out_dir)   # optional, for standalone inference
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoraConfig:
    """Everything defining a finetuning run. Serialized next to the adapter.

    Mirrors the convention in generation/generate.py's GenConfig: anything that defines
    the experiment is visible here, never buried mid-function.
    """

    model_id: str
    train_jsonl: Path
    adapter_dir: Path
    # Values left unset deliberately: research_context.md fixes none of these, and
    # guessing them here would pass off a default as a decision.
    num_layers: int
    batch_size: int
    iters: int
    learning_rate: float
    seed: int = 0


def train_lora(cfg: LoraConfig) -> dict:
    """Run LoRA finetuning and write the adapter plus provenance.

    Must write, next to the adapter: the resolved LoraConfig, the base model sha, the
    source run directory the training data came from, and that run's measured Ĥ(q) --
    without the last one the final loss cannot be read as a divergence.

    Args:
        cfg: the resolved finetuning configuration.

    Returns:
        {adapter_dir, final_train_loss, final_val_loss, n_supervised_tokens, provenance}.
    """
    raise NotImplementedError(
        "train_lora is not implemented. Unverified prerequisite: whether mlx_lm.lora "
        "trains a 4B 8-bit model within 16 GB on this machine at all. Undecided: the "
        "hyperparameters above, and whether to call mlx_lm.lora's Python API or shell "
        "out to its CLI -- the CLI is better supported but makes the completion mask "
        "from finetune/data.py harder to pass through explicitly."
    )


def fuse_adapter(adapter_dir: Path, out_dir: Path) -> Path:
    """Fuse a trained adapter into standalone weights for inference.

    Produces a directory that generation/generate.py can load with --model, so the
    finetuned model goes through exactly the same generation path as the base model. That
    identity is what makes evals/compare.py's before/after comparison valid.

    Args:
        adapter_dir: directory written by train_lora.
        out_dir: destination for the fused weights.

    Returns:
        Path to the fused model directory.
    """
    raise NotImplementedError(
        "fuse_adapter is not implemented. Depends on train_lora. Should write a "
        "download.json-compatible manifest into out_dir so that generation's provenance "
        "resolution reports the base sha and the adapter, rather than 'unresolved'."
    )
