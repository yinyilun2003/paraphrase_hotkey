import clipboard_io


class StrSubclass(str):
    """模拟 macOS 上 pyperclip 返回的 objc.pyobjc_unicode。"""


def test_read_text_returns_plain_str(monkeypatch):
    monkeypatch.setattr(clipboard_io.pyperclip, "paste", lambda: StrSubclass("原文"))
    text = clipboard_io.read_text()
    assert text == "原文"
    assert type(text) is str


def test_read_text_empty_is_none(monkeypatch):
    monkeypatch.setattr(clipboard_io.pyperclip, "paste", lambda: "")
    assert clipboard_io.read_text() is None
