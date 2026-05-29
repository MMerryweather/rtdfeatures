"""Blocked bootstrap utilities package."""

from rtdfeatures.bootstrap.contracts import BlockedBootstrapConfig, BootstrapIndexSplit
from rtdfeatures.bootstrap.sampling import (
    bootstrap_kernel_candidates,
    bootstrap_kernel_fit,
    generate_blocked_bootstrap_splits,
)
from rtdfeatures.bootstrap.summaries import (
    bootstrap_lag_interval_table,
    bootstrap_lag_summary_samples_table,
    bootstrap_parameter_interval_table,
    bootstrap_parameter_samples_table,
    bootstrap_summary_compact_dict,
    bootstrap_summary_compact_text,
    bootstrap_weight_interval_table,
    bootstrap_weight_samples_table,
    build_kernel_bootstrap_summary,
)

# Re-exported for backward compatibility with tests that used the old monolith
from rtdfeatures.diagnostics import (
    BOOTSTRAP_WARNING_CODES,
    DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES,
)

__all__ = [
    "BlockedBootstrapConfig",
    "BootstrapIndexSplit",
    "BOOTSTRAP_WARNING_CODES",
    "DEFAULT_BOOTSTRAP_INTERVAL_QUANTILES",
    "bootstrap_kernel_candidates",
    "bootstrap_kernel_fit",
    "bootstrap_lag_interval_table",
    "bootstrap_lag_summary_samples_table",
    "bootstrap_parameter_interval_table",
    "bootstrap_parameter_samples_table",
    "bootstrap_summary_compact_dict",
    "bootstrap_summary_compact_text",
    "bootstrap_weight_interval_table",
    "bootstrap_weight_samples_table",
    "build_kernel_bootstrap_summary",
    "generate_blocked_bootstrap_splits",
]
