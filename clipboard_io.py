"""读剪贴板、写剪贴板、模拟粘贴、恢复剪贴板（Windows / macOS）。

手动测试：python clipboard_io.py  —— 3 秒内切到记事本（macOS 用备忘录），会粘贴一行测试文字，随后恢复原剪贴板。
"""

from __future__ import annotations

import sys
import time

import pyperclip
from pynput.keyboard import Controller, Key, KeyCode

IS_MAC = sys.platform == "darwin"
if IS_MAC:
    import Quartz
else:
    import ctypes

VK_CONTROL = 0x11
# macOS 按键位发 V，不受键盘布局和输入法影响
_PASTE_KEYS = (Key.cmd, KeyCode.from_vk(0x09)) if IS_MAC else (Key.ctrl, "v")
SETTLE_DELAY_S = 0.08   # 触发后等剪贴板稳定
RESTORE_DELAY_S = 0.15  # 粘贴后等目标应用读完剪贴板再恢复

_keyboard = Controller()


def wait_settle(delay: float = SETTLE_DELAY_S) -> None:
    time.sleep(delay)


def read_text() -> str | None:
    """读剪贴板。失败、为空或非文本时返回 None。"""
    try:
        text = pyperclip.paste()
    except Exception:
        return None
    if not isinstance(text, str) or text == "":
        return None
    # macOS 上 pyperclip 返回 objc.pyobjc_unicode（str 的子类），google-genai 会把它序列化成空内容
    return str(text)


def write_text(s: str, retries: int = 3) -> None:
    """写剪贴板。剪贴板被其他进程占用时短暂重试，仍失败则抛出异常。"""
    for attempt in range(retries):
        try:
            pyperclip.copy(s)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.05)


def _ctrl_physically_down() -> bool:
    """Ctrl（macOS 上是 Command）是否被物理按住。"""
    if IS_MAC:
        flags = Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateHIDSystemState)
        return bool(flags & Quartz.kCGEventFlagMaskCommand)
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)


def _mac_front_window() -> int:
    """最前面的普通窗口（layer 0）的窗口号。只读窗口号，不需要“屏幕录制”权限。"""
    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    ) or []
    for w in windows:  # 按从前到后的顺序排列
        if w.get(Quartz.kCGWindowLayer) == 0:
            return int(w.get(Quartz.kCGWindowNumber, 0))
    return 0


def foreground_window() -> int:
    """当前前台窗口句柄（macOS 上是窗口号），用来确认粘贴时焦点没换到别的窗口。"""
    if IS_MAC:
        return _mac_front_window()
    return ctypes.windll.user32.GetForegroundWindow() or 0


def wait_ctrl_released(timeout: float = 2.0) -> bool:
    """等用户松开 Ctrl（macOS 上是 Command）。返回 True 表示已松开，False 表示超时。"""
    deadline = time.monotonic() + timeout
    while _ctrl_physically_down():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def paste() -> None:
    """模拟 Ctrl+V（macOS 上是 Cmd+V）。"""
    modifier, v = _PASTE_KEYS
    with _keyboard.pressed(modifier):
        _keyboard.press(v)
        _keyboard.release(v)


def restore(original: str, delay: float = RESTORE_DELAY_S) -> None:
    """等粘贴完成后把 original 写回剪贴板。"""
    time.sleep(delay)
    write_text(original)


def _demo() -> None:
    original = read_text()
    print(f"3 秒后会在当前焦点窗口粘贴测试文字，请切到{'备忘录' if IS_MAC else '记事本'}……")
    time.sleep(3)
    write_text("clipboard_io 测试：这行文字由程序粘贴。")
    if not wait_ctrl_released():
        print(f"{'Cmd' if IS_MAC else 'Ctrl'} 一直按着，已放弃粘贴")
    else:
        paste()
    if original is not None:
        restore(original)
        print("已粘贴，剪贴板已恢复为原内容")
    else:
        print("已粘贴（原剪贴板为空或非文本，未恢复）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _demo()
