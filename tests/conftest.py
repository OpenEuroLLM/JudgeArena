from contextlib import ExitStack
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_task_registries():
    from judgearena.benchmarks.scoring import _METRIC_TYPES
    from judgearena.datasets.registry import _DATASETS
    from judgearena.prompts.parsing import JUDGE_PARSERS
    from judgearena.prompts.registry import PRESETS
    from judgearena.utils.extensions import load_task_components

    # Register packaged components before taking the snapshots restored at teardown.
    load_task_components()
    with ExitStack() as stack:
        for registry in (PRESETS, JUDGE_PARSERS, _DATASETS, _METRIC_TYPES):
            stack.enter_context(patch.dict(registry))
        yield
