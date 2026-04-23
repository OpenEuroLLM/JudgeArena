import judgearena.utils as utils
from judgearena.utils import make_model


def test_download_all_dispatches_arena_hard_versions(monkeypatch, tmp_path):
    calls: list[tuple[str, str, object]] = []

    monkeypatch.setattr(utils, "data_root", tmp_path)
    monkeypatch.setattr(
        utils,
        "download_hf",
        lambda name, local_path: calls.append(("hf", name, local_path)),
    )
    monkeypatch.setattr(
        utils,
        "download_arena_hard",
        lambda dataset, local_tables_path: calls.append(
            ("arena", dataset, local_tables_path)
        ),
    )
    monkeypatch.setattr(
        utils,
        "snapshot_download",
        lambda **kwargs: calls.append(
            ("snapshot", kwargs["repo_id"], kwargs["local_dir"])
        ),
    )

    utils.download_all()

    tables_dir = tmp_path / "tables"
    assert calls[:4] == [
        ("hf", "alpaca-eval", tables_dir),
        ("arena", "arena-hard-v0.1", tables_dir),
        ("arena", "arena-hard-v2.0", tables_dir),
        ("hf", "m-arena-hard", tables_dir),
    ]
    assert calls[4] == (
        "snapshot",
        "geoalgo/multilingual-contexts-to-be-completed",
        tmp_path / "contexts",
    )


def test_make_model_openrouter_strips_vllm_only_kwargs(monkeypatch):
    """vLLM-engine-only kwargs must not leak into ChatOpenAI.model_kwargs.

    Regression guard for #20: unknown kwargs forwarded to ``ChatOpenAI`` land
    in ``model_kwargs`` and are then sent to ``chat.completions.create``,
    which rejects them with ``TypeError: unexpected keyword argument
    'max_model_len'``.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    model = make_model(
        "OpenRouter/google/gemma-3-4b-it",
        max_tokens=16,
        max_model_len=4096,
        chat_template="<ct>",
        temperature=0.5,
    )

    assert "max_model_len" not in model.model_kwargs
    assert "chat_template" not in model.model_kwargs
    assert model.max_tokens == 16
    assert model.temperature == 0.5
