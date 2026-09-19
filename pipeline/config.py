"""Configuration is a plain nested dict, loaded from a YAML file.

    cfg = load_config("configs/rec.yaml")
    cfg["eval"]["ks"]                       # -> [5, 10, 20]
    cfg["model"]["params"]["epochs"] = 2    # edit it like any dict
    train(cfg)                              # the stages accept the dict directly

`configs/main.yaml` is the single list of sections, keys and default values. Anything a config leaves
out falls back to it, and keys that it does not list are rejected so that typos do not pass silently.
Model hyper-parameters are the exception: `model.params` is defined by each model (its `DEFAULTS`).
"""

import re
from functools import cache
from pathlib import Path
import yaml

MAIN_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "main.yaml"


class _Loader(yaml.SafeLoader):
    """SafeLoader that reads `2e-4` as a float (PyYAML's YAML 1.1 rules need `2.0e-4`)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*\.[0-9_]*|\.[0-9_]+|[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)?$|^[-+]?\.(?:inf|Inf|INF)$|^\.(?:nan|NaN|NAN)$"),
    list("-+0123456789."),
)


def _read(path: Path) -> dict:
    if path.suffix not in (".yaml", ".yml"):
        raise SystemExit(f"Config must be a .yaml file, got {path.name!r}")
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader) or {}


@cache
def defaults() -> dict[str, dict]:
    """Sections, keys and default values from `configs/main.yaml`."""
    return _read(MAIN_CONFIG)


def _snake(d: dict) -> dict:
    """`wandb-project` and `wandb_project` are the same key."""
    return {k.replace("-", "_"): v for k, v in d.items()}


def with_defaults(user: dict | None, defaults: dict, where: str, required: tuple[str, ...] = ()) -> dict:
    """`defaults` overridden by `user`; unknown keys and missing `required` keys stop the run."""
    user = _snake(user or {})
    if unknown := sorted(user.keys() - defaults.keys() - set(required)):
        raise SystemExit(f"Unknown keys in [{where}]: {unknown}; allowed: {sorted(defaults.keys() | set(required))}")
    if missing := [k for k in required if k not in user]:
        raise SystemExit(f"[{where}] is missing required keys: {missing}")
    return {**defaults, **user}


def resolve(cfg: dict) -> dict:
    """Fill in defaults for every section. Safe to call on an already resolved config."""
    base, cfg = defaults(), _snake(cfg)
    if unknown := sorted(k for k in cfg.keys() - base.keys() if not k.startswith("_")):
        raise SystemExit(f"Unknown config sections: {unknown}; allowed: {sorted(base)}")
    out = {name: with_defaults(cfg.get(name), section, name) for name, section in base.items()}
    return out | {k: v for k, v in cfg.items() if k.startswith("_")}  # bookkeeping such as `_config_path`


def require(cfg: dict, section: str, *keys: str) -> None:
    """Stop with a clear message if `[section]` leaves out keys that have no usable default."""
    if missing := [k for k in keys if cfg[section].get(k) is None]:
        raise SystemExit(f"[{section}] is missing required keys: {missing}")



def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result

def load_config(path: str) -> dict:
    """Load main.yaml and recursively override it with the supplied config."""
    config_path = Path(path)

    main = _read(MAIN_CONFIG)
    override = _read(config_path)

    config = _deep_merge(main, override)
    config["_config_path"] = str(config_path.resolve())

    return config