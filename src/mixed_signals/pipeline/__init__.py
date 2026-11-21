"""Config-driven pipeline: a YAML config, a runner and a run folder.

A config names the posteriors to read, the parameter space and the engines to run
(``diagnose``, ``compare`` and ``compare_signals``). The pipeline reads the data,
estimates the cost of every engine, runs them with one seed and writes a run
folder with the results, the hashed config, the reductions of the chains, the
estimate and the provenance. :mod:`mixed_signals.pipeline.config` is the
reference of the config format, and :mod:`mixed_signals.pipeline.runner`
describes the run and its files. The command ``mixed-signals run config.yaml``
runs the same function.

A run from Python::

    from mixed_signals.pipeline import run_pipeline

    run = run_pipeline("config.yaml")
    run.result.table  # polars DataFrame of every row
    run.output  # the run folder

The pipeline is not re-exported by ``import mixed_signals``, which keeps the
import time of the package unchanged.

A config runs the code that it names by import path, like a script: run configs
that you trust.
"""

from mixed_signals.pipeline.config import PipelineConfig, load_config
from mixed_signals.pipeline.runner import (
    PipelineEstimate,
    PipelineRun,
    estimate_pipeline,
    run_pipeline,
)

__all__ = [
    "PipelineConfig",
    "PipelineEstimate",
    "PipelineRun",
    "estimate_pipeline",
    "load_config",
    "run_pipeline",
]
