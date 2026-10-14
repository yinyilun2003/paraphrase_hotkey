import pytest

from config import ConfigError, load_config

KEY = "test-key-should-never-leak-123"


def test_defaults():
    cfg = load_config({"GEMINI_API_KEY": KEY})
    assert cfg.api_key == KEY
    assert cfg.model == "gemini-3.1-flash-lite"
    assert cfg.double_press_window_ms == 400
    assert cfg.max_input_chars == 4000
    assert cfg.request_timeout_s == 10.0
    assert cfg.log_text is False


def test_overrides():
    cfg = load_config({
        "GEMINI_API_KEY": KEY,
        "GEMINI_MODEL": "some-model",
        "DOUBLE_PRESS_WINDOW_MS": "300",
        "MAX_INPUT_CHARS": "100",
        "REQUEST_TIMEOUT_S": "2.5",
        "LOG_TEXT": "TRUE",
    })
    assert cfg.model == "some-model"
    assert cfg.double_press_window_ms == 300
    assert cfg.max_input_chars == 100
    assert cfg.request_timeout_s == 2.5
    assert cfg.log_text is True


def test_empty_values_fall_back_to_defaults():
    cfg = load_config({"GEMINI_API_KEY": KEY, "GEMINI_MODEL": "", "MAX_INPUT_CHARS": " "})
    assert cfg.model == "gemini-3.1-flash-lite"
    assert cfg.max_input_chars == 4000


@pytest.mark.parametrize("env", [{}, {"GEMINI_API_KEY": ""}, {"GEMINI_API_KEY": "   "}])
def test_missing_key(env):
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_config(env)


@pytest.mark.parametrize("name,value", [
    ("DOUBLE_PRESS_WINDOW_MS", "abc"),
    ("DOUBLE_PRESS_WINDOW_MS", "0"),
    ("DOUBLE_PRESS_WINDOW_MS", "-1"),
    ("DOUBLE_PRESS_WINDOW_MS", "1.5"),
    ("MAX_INPUT_CHARS", "lots"),
    ("REQUEST_TIMEOUT_S", "0"),
    ("REQUEST_TIMEOUT_S", "nan"),
    ("REQUEST_TIMEOUT_S", "inf"),
    ("LOG_TEXT", "maybe"),
])
def test_invalid_values(name, value):
    with pytest.raises(ConfigError, match=name) as exc:
        load_config({"GEMINI_API_KEY": KEY, name: value})
    assert KEY not in str(exc.value)


def test_repr_hides_key():
    cfg = load_config({"GEMINI_API_KEY": KEY})
    assert KEY not in repr(cfg)
    assert KEY not in str(cfg)
