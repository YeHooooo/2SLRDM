from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _convert(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _convert(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_convert(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as f:
        return _convert(yaml.safe_load(f))


def to_plain_dict(config: Config | dict) -> dict:
    return deepcopy(dict(config))

