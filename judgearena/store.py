"""Shared completion and judgement store backed by a HuggingFace dataset repo.

Layout
------
  completions/{task}/{model_name}/{provider}/{date}/
      metadata.json
      part-{HHMMSS}.parquet

  judgements/{task}/{judge_name}/{provider}/{date}/
      metadata.json
      part-{HHMMSS}.parquet

Example: model "OpenRouter/meta-llama/Llama-3.3-70B-Instruct" on 2026-07-02 →
  completions/alpaca-eval/meta-llama--Llama-3.3-70B-Instruct/OpenRouter/20260702/
      metadata.json
      part-153045.parquet

Write path: every save() creates a new parquet file timestamped to the second.
Read path:  load() downloads all matching shards and deduplicates in pandas;
            query() then filters the in-memory data without hitting the network.
"""

import io
import json
import uuid
from datetime import UTC, datetime

import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from judgearena.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_model_spec(model_spec: str) -> tuple[str, str]:
    """Split 'Provider/org/model' into (provider, 'org--model')."""
    parts = model_spec.split("/")
    provider = parts[0]
    model_name = "--".join(parts[1:])
    return provider, model_name


def _model_folder(task: str, model_spec: str, date: str) -> str:
    provider, model_name = _parse_model_spec(model_spec)
    return f"completions/{task}/{model_name}/{provider}/{date}"


def _judge_folder(task: str, judge_spec: str, date: str) -> str:
    provider, judge_name = _parse_model_spec(judge_spec)
    return f"judgements/{task}/{judge_name}/{provider}/{date}"


def _today() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def _time_now() -> str:
    return datetime.now(UTC).strftime("%H%M%S")


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _metadata_operation(
    *,
    config: dict,
    folder: str,
    pushed_by: str,
    existing_files: set[str],
) -> CommitOperationAdd | None:
    """Return a CommitOperationAdd for metadata.json, or None if already present."""
    path = f"{folder}/metadata.json"
    if path in existing_files:
        return None
    payload = {
        **config,
        "pushed_by": pushed_by,
        "pushed_at": datetime.now(UTC).isoformat(),
    }
    existing_files.add(path)
    return CommitOperationAdd(
        path_in_repo=path,
        path_or_fileobj=json.dumps(payload, sort_keys=True, indent=2).encode(),
    )


def _download_shards(
    api: HfApi, *, hf_repo: str, prefix: str, cache_dir: str | None
) -> pd.DataFrame:
    files = [
        f
        for f in api.list_repo_files(repo_id=hf_repo, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    ]
    if not files:
        return pd.DataFrame()
    dfs = []
    for hf_path in files:
        local = hf_hub_download(
            repo_id=hf_repo,
            repo_type="dataset",
            filename=hf_path,
            cache_dir=cache_dir,
        )
        dfs.append(pd.read_parquet(local))
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# CompletionStore
# ---------------------------------------------------------------------------


class CompletionStore:
    """Store and retrieve model completions on a shared HF dataset repo.

    Usage::

        store = CompletionStore(hf_repo="myorg/store", task="alpaca-eval")
        store.save(df, model_config={"model": "OpenRouter/meta-llama/Llama-3.3-70B-Instruct"},
                   pushed_by="alice")
        store.load(model_config={"model": "OpenRouter/meta-llama/Llama-3.3-70B-Instruct"})
        df = store.query(model="meta-llama/Llama-3.3-70B-Instruct")
    """

    def __init__(self, hf_repo: str, task: str) -> None:
        self.hf_repo = hf_repo
        self.task = task
        self._df: pd.DataFrame | None = None

    def save(
        self,
        df: pd.DataFrame,
        model_config: dict,
        pushed_by: str,
        run_id: str | None = None,
        create_pr: bool = False,
    ) -> str:
        """Push completions to the store.

        Args:
            df: DataFrame with at minimum ``instruction_index`` and ``completion``.
            model_config: Dict with ``model`` key plus inference hyperparams.
                Hyperparams are stored in metadata.json; only ``model`` affects the path.
            pushed_by: HF username or job identifier.
            run_id: UUID for this batch; auto-generated if omitted.
            create_pr: If True, open a pull request instead of pushing directly.
                Use when you don't have write access to the repo.

        Returns:
            PR URL when ``create_pr=True``, otherwise path-in-repo of the uploaded file.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        api = HfApi()
        existing = set(api.list_repo_files(repo_id=self.hf_repo, repo_type="dataset"))

        folder = _model_folder(self.task, model_config["model"], _today())
        hf_path = f"{folder}/part-{_time_now()}.parquet"

        df = df.copy()
        df["pushed_by"] = pushed_by
        df["pushed_at"] = datetime.now(UTC).isoformat()
        df["run_id"] = run_id
        parquet_bytes = _to_parquet_bytes(df)

        operations = []
        meta_op = _metadata_operation(
            config=model_config,
            folder=folder,
            pushed_by=pushed_by,
            existing_files=existing,
        )
        if meta_op is not None:
            operations.append(meta_op)
        operations.append(
            CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=parquet_bytes)
        )

        commit_info = api.create_commit(
            repo_id=self.hf_repo,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add completions {self.task}/{model_config['model']} run={run_id}",
            create_pr=create_pr,
        )
        if create_pr:
            logger.info("Opened PR for %d completions: %s", len(df), commit_info.pr_url)
            return commit_info.pr_url
        logger.info("Pushed %d completions to %s", len(df), hf_path)
        return hf_path

    def load(
        self,
        model_config: dict | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Fetch shards from the store into memory and deduplicate.

        Args:
            model_config: If given, fetch only this model's shards (all dates).
            cache_dir: Optional local HF cache directory.
        """
        api = HfApi()
        if model_config is not None:
            provider, model_name = _parse_model_spec(model_config["model"])
            prefix = f"completions/{self.task}/{model_name}/{provider}/"
        else:
            prefix = f"completions/{self.task}/"

        df = _download_shards(
            api, hf_repo=self.hf_repo, prefix=prefix, cache_dir=cache_dir
        )
        if not df.empty and "pushed_at" in df.columns:
            df = (
                df.sort_values("pushed_at", ascending=False)
                .drop_duplicates(subset=["instruction_index"])
                .sort_values("instruction_index")
                .reset_index(drop=True)
            )
        self._df = df

    def query(self, model: str | None = None) -> pd.DataFrame:
        """Filter the in-memory data loaded by :meth:`load`.

        Args:
            model: If given, keep only rows where ``model`` column equals this value.

        Returns:
            Filtered DataFrame.
        """
        if self._df is None:
            raise RuntimeError("Call load() before query().")
        df = self._df
        if model is not None and "model" in df.columns:
            df = df[df["model"] == model]
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# JudgementStore
# ---------------------------------------------------------------------------


class JudgementStore:
    """Store and retrieve pairwise LLM judge annotations on a shared HF dataset repo.

    Each row has its own ``model_A`` and ``model_B`` string identifiers.
    The evaluated model can appear in either position.

    Usage::

        store = JudgementStore(hf_repo="myorg/store", task="alpaca-eval")
        # df columns: instruction_index, model_A, model_B, judge_input, judge_output
        store.save(df, judge_config={"model": "OpenRouter/meta-llama/Llama-3.3-70B-Instruct"},
                   pushed_by="alice")
        store.load()
        df = store.query(model="my-model")
    """

    def __init__(self, hf_repo: str, task: str) -> None:
        self.hf_repo = hf_repo
        self.task = task
        self._df: pd.DataFrame | None = None

    def save(
        self,
        df: pd.DataFrame,
        judge_config: dict,
        pushed_by: str,
        run_id: str | None = None,
        create_pr: bool = False,
    ) -> str:
        """Push judgements to the store.

        Args:
            df: DataFrame with columns ``instruction_index``, ``model_A``,
                ``model_B``, ``judge_input``, ``judge_output``.
            judge_config: Dict with ``model`` key plus judge hyperparams.
                Hyperparams are stored in metadata.json; only ``model`` affects the path.
            pushed_by: HF username or job identifier.
            run_id: UUID for this batch; auto-generated if omitted.
            create_pr: If True, open a pull request instead of pushing directly.
                Use when you don't have write access to the repo.

        Returns:
            PR URL when ``create_pr=True``, otherwise path-in-repo of the uploaded file.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        api = HfApi()
        existing = set(api.list_repo_files(repo_id=self.hf_repo, repo_type="dataset"))

        folder = _judge_folder(self.task, judge_config["model"], _today())
        hf_path = f"{folder}/part-{_time_now()}.parquet"

        df = df.copy()
        df["pushed_by"] = pushed_by
        df["pushed_at"] = datetime.now(UTC).isoformat()
        df["run_id"] = run_id
        parquet_bytes = _to_parquet_bytes(df)

        operations = []
        meta_op = _metadata_operation(
            config=judge_config,
            folder=folder,
            pushed_by=pushed_by,
            existing_files=existing,
        )
        if meta_op is not None:
            operations.append(meta_op)
        operations.append(
            CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=parquet_bytes)
        )

        commit_info = api.create_commit(
            repo_id=self.hf_repo,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add judgements {self.task}/{judge_config['model']} run={run_id}",
            create_pr=create_pr,
        )
        if create_pr:
            logger.info("Opened PR for %d judgements: %s", len(df), commit_info.pr_url)
            return commit_info.pr_url
        logger.info("Pushed %d judgements to %s", len(df), hf_path)
        return hf_path

    def load(
        self,
        judge_config: dict | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Fetch shards from the store into memory and deduplicate.

        Args:
            judge_config: If given, fetch only this judge's shards (all dates).
            cache_dir: Optional local HF cache directory.
        """
        api = HfApi()
        if judge_config is not None:
            provider, judge_name = _parse_model_spec(judge_config["model"])
            prefix = f"judgements/{self.task}/{judge_name}/{provider}/"
        else:
            prefix = f"judgements/{self.task}/"

        df = _download_shards(
            api, hf_repo=self.hf_repo, prefix=prefix, cache_dir=cache_dir
        )
        if not df.empty and "pushed_at" in df.columns:
            df = (
                df.sort_values("pushed_at", ascending=False)
                .drop_duplicates(subset=["instruction_index", "model_A", "model_B"])
                .sort_values(["instruction_index", "model_A", "model_B"])
                .reset_index(drop=True)
            )
        self._df = df

    def query(
        self,
        model_A: str | None = None,
        model_B: str | None = None,
        model: str | None = None,
    ) -> pd.DataFrame:
        """Filter the in-memory data loaded by :meth:`load`.

        Args:
            model_A: Keep only rows where position A is this model.
            model_B: Keep only rows where position B is this model.
            model: Keep rows where this model appears in either position.
                   Cannot be combined with ``model_A`` / ``model_B``.

        Returns:
            Filtered DataFrame.
        """
        if self._df is None:
            raise RuntimeError("Call load() before query().")
        if model is not None and (model_A is not None or model_B is not None):
            raise ValueError("Use either `model` or `model_A`/`model_B`, not both.")

        df = self._df
        if model is not None:
            df = df[(df["model_A"] == model) | (df["model_B"] == model)]
        if model_A is not None:
            df = df[df["model_A"] == model_A]
        if model_B is not None:
            df = df[df["model_B"] == model_B]
        return df.reset_index(drop=True)
