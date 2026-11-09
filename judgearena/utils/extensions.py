"""Load Python components and task-local registrations."""

import hashlib
import importlib.util
import sys
from pathlib import Path


def python_reference(name: str | None) -> tuple[Path, str] | None:
    """Split an explicit Python file reference; plain names remain registry keys."""
    if not isinstance(name, str):
        return None
    filename, separator, symbol = name.rpartition(":")
    if not separator or not filename.endswith(".py"):
        return None
    if not symbol.isidentifier():
        raise ValueError(f"Python reference {name!r} requires a Python object name.")
    return Path(filename), symbol


def load_python_module(path: Path):
    path = path.resolve()
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read Python component {path}: {exc}") from exc
    digest = hashlib.sha256(str(path).encode() + b"\0" + source).hexdigest()
    module_name = f"_judgearena_extension_{digest}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            # Compile the read source: same-size edits can leave timestamp-based pyc stale.
            exec(compile(source, str(path), "exec"), module.__dict__)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    return module


_object_references = {}
_loaded_registrations = set()
_loading_registrations = False


def load_python_object(path: Path, symbol: str):
    obj = getattr(load_python_module(path), symbol, None)
    if obj is not None:
        _object_references[id(obj)] = f"{path.resolve()}:{symbol}"
    return obj


def python_object_reference(obj):
    """Return the explicit file export used to load this object, when known."""
    return _object_references.get(id(obj))


def load_task_components(root=None):
    """Import each task family's register.py once, before resolving public names."""
    global _loading_registrations
    if _loading_registrations:
        return
    root = (
        Path(root)
        if root is not None
        else Path(__file__).resolve().parents[1] / "tasks/definitions"
    )
    _loading_registrations = True
    try:
        for path in sorted(root.glob("*/register.py")):
            path = path.resolve()
            if path not in _loaded_registrations:
                load_python_module(path)
                _loaded_registrations.add(path)
    finally:
        _loading_registrations = False


def config_definitions(data):
    """Yield the effective configuration and any saved task definition."""
    yield data
    resolved = data.get("resolved_task")
    if isinstance(resolved, dict) and isinstance(resolved.get("spec"), dict):
        yield resolved["spec"]


def component_references(data):
    """Yield adapter and metric fields that can refer to Python implementations."""
    for definition in config_definitions(data):
        dataset = definition.get("dataset")
        if isinstance(dataset, dict) and "adapter" in dataset:
            yield dataset, "adapter"
        protocol = definition.get("protocol")
        scoring = protocol.get("scoring") if isinstance(protocol, dict) else None
        metrics = scoring.get("metrics") if isinstance(scoring, dict) else None
        if isinstance(metrics, (list, tuple)):
            for metric in metrics:
                if isinstance(metric, dict) and "metric" in metric:
                    yield metric, "metric"


def resolve_component_paths(data, directory):
    for owner, key in component_references(data):
        reference = python_reference(owner[key])
        if reference is not None:
            path, symbol = reference
            if not path.is_absolute():
                owner[key] = f"{directory / path}:{symbol}"
