"""Reproducible panel selection without model inference."""

import pandas as pd
import yaml
from test_leaderboard_cli import setup_path as setup_path

import judgearena.models as models
from judgearena.cli import cli


def test_panel_selection_reuses_the_seed(setup_path, tmp_path, monkeypatch):
    def no_inference(*_args, **_kwargs):
        raise AssertionError("Creating a fixed-beta benchmark must not call a model")

    monkeypatch.setattr(models, "make_model", no_inference)
    setup = yaml.safe_load(setup_path.read_text())
    panels = []
    for index, seed in enumerate((7, 7, 19)):
        setup["evaluation"]["run"]["seed"] = seed
        setup_path.write_text(yaml.safe_dump(setup))
        directory = tmp_path / f"board-{index}"
        cli(["leaderboard", "create", str(setup_path), "--output", str(directory)])
        panels.append(pd.read_parquet(directory / "panel.parquet"))

    pd.testing.assert_frame_equal(panels[0], panels[1])
    assert not panels[0].equals(panels[2])
