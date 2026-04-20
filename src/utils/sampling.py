"""Sampling backend helpers for PyMC/Bambi."""

from __future__ import annotations

import os


def _safe_cores(chains: int) -> int:
    """Choose a conservative process count for multi-chain sampling."""
    return max(1, min(chains, os.cpu_count() or 1))


def _cuda_available() -> bool:
    """Return True if JAX can see a GPU device (requires jax[cuda12_pip] + numpyro)."""
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def get_bambi_sampler_kwargs(chains: int) -> dict:
    """Use numpyro (JAX/CUDA) when a GPU is available, else PyMC CPU."""
    if _cuda_available():
        return {"inference_method": "nuts_numpyro"}
    return {"inference_method": "pymc", "cores": _safe_cores(chains)}


def get_pymc_sampler_kwargs(chains: int) -> dict:
    """Stable PyMC sampler config."""
    if _cuda_available():
        return {"nuts_sampler": "numpyro"}
    return {"cores": _safe_cores(chains)}
