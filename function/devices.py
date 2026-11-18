"""Explicit inference device selection, independent of training configuration."""
import torch


def resolve_inference_device(requested="cpu"):
    """Resolve cpu/cuda/auto; reject unavailable CUDA rather than silently falling back."""
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("Inference device must be cpu, cuda, or auto.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu.")
    return torch.device(requested)
