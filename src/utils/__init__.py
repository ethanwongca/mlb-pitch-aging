"""Utility helpers for the project."""

from .utils import BASE_DIR
from .utils import DATA_PATH
from .utils import DEFAULT_OUTCOMES
from .utils import OUTCOME_LABELS
from .utils import PITCH_COLORS
from .utils import PITCH_LABELS
from .utils import get_age_mean
from .utils import get_data_pitch_type_dict
from .utils import get_default_outcomes
from .utils import ensure_mean_pfx_x_norm
from .utils import filter_pitchers_by_min_distinct_seasons
from .utils import get_valid_pitch_types
from .utils import load_data
from .utils import setup_logger
from .sampling import get_bambi_sampler_kwargs
from .sampling import get_pymc_sampler_kwargs
from .inference import check_convergence
from .plots import plot_aging_curves_grid
from .plots import plot_decline_rate_ci
from .plots import plot_delta_method_comparison
from .plots import plot_ext_loo_heatmap
from .plots import plot_pareto_k_heatmap
from .plots import plot_peak_age_ci
from .plots import plot_bivariate_correlation
from .plots import plot_bivariate_peak_comparison
from .plots import plot_scg_bars
from .plots import plot_scg_dumbbell
from .plots import plot_spin_velo_divergence
from .plots import plot_spaghetti
from .plots import plot_survivorship_bias

__all__ = [
    "BASE_DIR",
    "DATA_PATH",
    "DEFAULT_OUTCOMES",
    "OUTCOME_LABELS",
    "PITCH_COLORS",
    "PITCH_LABELS",
    "load_data",
    "get_data_pitch_type_dict",
    "get_valid_pitch_types",
    "get_default_outcomes",
    "get_age_mean",
    "ensure_mean_pfx_x_norm",
    "filter_pitchers_by_min_distinct_seasons",
    "setup_logger",
    "get_bambi_sampler_kwargs",
    "get_pymc_sampler_kwargs",
    "check_convergence",
    "plot_aging_curves_grid",
    "plot_decline_rate_ci",
    "plot_delta_method_comparison",
    "plot_ext_loo_heatmap",
    "plot_pareto_k_heatmap",
    "plot_peak_age_ci",
    "plot_bivariate_correlation",
    "plot_bivariate_peak_comparison",
    "plot_scg_bars",
    "plot_scg_dumbbell",
    "plot_spin_velo_divergence",
    "plot_survivorship_bias",
    "plot_spaghetti",
]
