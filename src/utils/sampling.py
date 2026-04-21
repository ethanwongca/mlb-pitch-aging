"""Sampling backend helpers for PyMC/Bambi."""

from __future__ import annotations

import os


def _safe_cores(chains: int) -> int:
    """Choose a conservative process count for multi-chain sampling."""
    return max(1, min(chains, os.cpu_count() or 1))


def get_bambi_sampler_kwargs(chains: int) -> dict:
    return {"inference_method": "pymc", "cores": _safe_cores(chains)}


def get_pymc_sampler_kwargs(chains: int) -> dict:
    return {"cores": _safe_cores(chains)}
