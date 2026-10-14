"""读取并校验 .env 配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"


class ConfigError(Exception):
    """配置缺失或非法。错误信息里不包含 key 本身。"""


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str = "gemini-3.1-flash-lite"
    double_press_window_ms: int = 400
    max_input_chars: int = 4000
    request_timeout_s: float = 10.0
    log_text: bool = False

    def __repr__(self) -> str:  # 防止 key 被意外打印
        return (
            f"Config(api_key=***, model={self.model!r}, "
            f"double_press_window_ms={self.double_press_window_ms}, "
            f"max_input_chars={self.max_input_chars}, "
            f"request_timeout_s={self.request_timeout_s}, log_text={self.log_text})"
        )

    __str__ = __repr__


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, "")
    value = value.strip() if value else ""
    return value or default


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _get(env, name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} 必须是正整数，当前值为 {raw!r}") from None
    if value <= 0:
        raise ConfigError(f"{name} 必须是正整数，当前值为 {raw!r}")
    return value


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _get(env, name, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} 必须是正数，当前值为 {raw!r}") from None
    if not value > 0 or value == float("inf"):
        raise ConfigError(f"{name} 必须是正数，当前值为 {raw!r}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name, "true" if default else "false").lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} 必须是 true 或 false，当前值为 {raw!r}")


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """从环境变量（默认先加载 .env）构造 Config。

    传入 env 时只使用该映射，不读 .env，便于测试。
    """
    if env is None:
        load_dotenv(ENV_PATH)
        env = os.environ

    api_key = _get(env, "GEMINI_API_KEY", "")
    if not api_key:
        raise ConfigError("GEMINI_API_KEY 未设置，请在 .env 中填写（参考 .env.example）")

    return Config(
        api_key=api_key,
        model=_get(env, "GEMINI_MODEL", Config.model),
        double_press_window_ms=_positive_int(env, "DOUBLE_PRESS_WINDOW_MS", Config.double_press_window_ms),
        max_input_chars=_positive_int(env, "MAX_INPUT_CHARS", Config.max_input_chars),
        request_timeout_s=_positive_float(env, "REQUEST_TIMEOUT_S", Config.request_timeout_s),
        log_text=_bool(env, "LOG_TEXT", Config.log_text),
    )
