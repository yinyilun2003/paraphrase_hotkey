"""main.App 的流程测试：用假剪贴板 / 假 paraphrase，不碰真实键盘和网络。"""

import threading

import pytest

import main
from config import Config
from paraphrase import ParaphraseError


class FakeClipboard:
    def __init__(self, text, ctrl_released=True, hwnds=(1, 1)):
        self.text = text
        self.ctrl_released = ctrl_released
        self.hwnds = list(hwnds)
        self.events = []

    def foreground_window(self):
        return self.hwnds.pop(0) if self.hwnds else 1

    def wait_settle(self):
        pass

    def read_text(self):
        return self.text

    def write_text(self, s):
        self.events.append(("write", s))
        self.text = s

    def wait_ctrl_released(self, timeout=2.0):
        return self.ctrl_released

    def paste(self):
        self.events.append(("paste", self.text))

    def restore(self, original):
        self.events.append(("restore", original))
        self.text = original


@pytest.fixture
def env(monkeypatch):
    beeps = []
    monkeypatch.setattr(main, "_beep", lambda: beeps.append(1))

    def setup(clip_text, result=None, error=None, cfg=None, **clip_kw):
        clip = FakeClipboard(clip_text, **clip_kw)
        calls = []

        def fake_paraphrase(text, lang, cfg=None):
            calls.append(text)
            app.langs.append(lang)
            if error:
                raise error
            return result

        monkeypatch.setattr(main, "clipboard_io", clip)
        monkeypatch.setattr(main, "paraphrase", fake_paraphrase)
        app = main.App(cfg or Config(api_key="k"))
        app.langs = []
        return app, clip, calls, beeps

    return setup


def test_success_pastes_then_restores_original(env):
    app, clip, calls, beeps = env("嗯 那个 我们明天 sync 一下", result="我们明天 sync 一下。")
    app.on_trigger()
    assert calls == ["嗯 那个 我们明天 sync 一下"]
    assert clip.events == [
        ("write", "我们明天 sync 一下。"),
        ("paste", "我们明天 sync 一下。"),
        ("restore", "嗯 那个 我们明天 sync 一下"),
    ]
    assert beeps == []
    assert len(app.durations_ms) == 1


@pytest.mark.parametrize("text", [None, "", "   \n\t"])
def test_empty_clipboard_does_nothing(env, text):
    app, clip, calls, beeps = env(text, result="x")
    app.on_trigger()
    assert calls == [] and clip.events == [] and beeps == []


def test_too_long_skipped_without_calling_gemini(env):
    app, clip, calls, beeps = env("a" * 11, result="x", cfg=Config(api_key="k", max_input_chars=10))
    app.on_trigger()
    assert calls == [] and clip.events == [] and beeps == [1]


@pytest.mark.parametrize("kind", ["network", "auth", "rate_limit", "blocked", "empty", "api"])
def test_paraphrase_failure_leaves_clipboard_untouched(env, kind):
    app, clip, calls, beeps = env("原文", error=ParaphraseError(kind, "失败"))
    app.on_trigger()
    assert clip.events == [] and clip.text == "原文" and beeps == [1]


def test_unchanged_result_not_pasted(env):
    app, clip, calls, beeps = env("  Done.  ", result="Done.")
    app.on_trigger()
    assert clip.events == [] and beeps == []


def test_ctrl_held_not_pasted(env):
    app, clip, calls, beeps = env("原文", result="改写", ctrl_released=False)
    app.on_trigger()
    assert clip.events == [] and beeps == [1]


def test_window_switched_not_pasted(env):
    app, clip, calls, beeps = env("原文", result="改写", hwnds=(1, 2))
    app.on_trigger()
    assert clip.events == [] and beeps == [1]


def test_unexpected_error_is_contained_and_lock_released(env, monkeypatch):
    app, clip, calls, beeps = env("原文", error=RuntimeError("boom"))
    app.on_trigger()
    assert beeps == [1]
    assert app._busy.acquire(blocking=False)  # 锁已释放


def test_reentrant_trigger_ignored(env, monkeypatch):
    started, release = threading.Event(), threading.Event()
    app, clip, calls, beeps = env("原文", result="改写")

    def slow(text, lang, cfg=None):
        calls.append(text)
        started.set()
        release.wait(2)
        return "改写"

    monkeypatch.setattr(main, "paraphrase", slow)
    t = threading.Thread(target=app.on_trigger)
    t.start()
    assert started.wait(2)
    app.on_trigger()  # 第二次触发，应被忽略
    release.set()
    t.join(2)
    assert calls == ["原文"]
    assert [e[0] for e in clip.events] == ["write", "paste", "restore"]


def test_letter_c_uses_chinese(env):
    app, clip, calls, beeps = env("please review", result="请审阅")
    app.on_trigger("c")
    assert app.langs == ["zh"]


def test_letter_e_uses_english_same_flow(env):
    app, clip, calls, beeps = env("请审阅一下", result="Please review.")
    app.on_trigger("e")
    assert app.langs == ["en"]
    assert clip.events == [
        ("write", "Please review."),
        ("paste", "Please review."),
        ("restore", "请审阅一下"),
    ]


def test_summary():
    app = main.App(Config(api_key="k"))
    assert "没有" in app.summary()
    app.durations_ms = [100, 200, 300, 400, 1000]
    s = app.summary()
    assert "p50=300" in s and "p95=" in s
