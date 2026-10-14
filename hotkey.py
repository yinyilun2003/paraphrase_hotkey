"""连按热键检测：Ctrl+C 接 Ctrl+C → "c"（中文），Ctrl+C 接 Ctrl+E → "e"（英文）。

DoublePressDetector 只处理抽象事件和时间戳，便于单元测试；
HotkeyListener 在 Windows 低级键盘钩子的过滤回调里同步驱动它，并在工作线程里执行 on_trigger(letter)。

按键只观察、不拦截，唯一的例外是组合里的那一下 Ctrl+E：它会被吞掉，
免得目标应用执行自己的 Ctrl+E 功能（跳到搜索框、居中等）。单独按 Ctrl+E 不受影响。

手动测试：python hotkey.py  —— 在其他窗口按 Ctrl+C C / Ctrl+C E 会打印 triggered c / e；在本窗口按 Ctrl+C 退出。
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from pynput import keyboard

CTRL = "ctrl"
C = "c"
E = "e"
OTHER = "other"

_VK_KIND = {0x11: CTRL, 0xA2: CTRL, 0xA3: CTRL, 0x43: C, 0x45: E}
_PRESS_MESSAGES = (0x0100, 0x0104)    # WM_KEYDOWN, WM_SYSKEYDOWN
_RELEASE_MESSAGES = (0x0101, 0x0105)  # WM_KEYUP, WM_SYSKEYUP
_LLKHF_INJECTED = 0x10 | 0x02         # LLKHF_INJECTED | LLKHF_LOWER_IL_INJECTED


class DoublePressDetector:
    """状态机：一次有效 Ctrl+C 之后 window_s 内再按 Ctrl+C 返回 "c"，再按 Ctrl+E 返回 "e"。"""

    def __init__(self, window_ms: int):
        self.window_s = window_ms / 1000.0
        self.ctrl_is_down = False
        self.letters_down: set[str] = set()  # 防长按重复
        self.last_ctrl_c_ts: float | None = None

    def press(self, kind: str, ts: float) -> str | None:
        """处理一次按下事件，触发时返回 "c" / "e"，否则返回 None。"""
        if kind == CTRL:
            self.ctrl_is_down = True
            return None
        if kind not in (C, E):
            # 中间按了别的键，就不再算连按
            self.last_ctrl_c_ts = None
            return None

        if kind in self.letters_down:  # 长按的自动重复
            return None
        self.letters_down.add(kind)
        if not self.ctrl_is_down:
            self.last_ctrl_c_ts = None
            return None

        if self.last_ctrl_c_ts is not None and ts - self.last_ctrl_c_ts <= self.window_s:
            self.last_ctrl_c_ts = None  # 清零，防止三连击触发两次
            return kind
        # 第一下必须是 Ctrl+C；超时的 Ctrl+E 只是普通按键
        self.last_ctrl_c_ts = ts if kind == C else None
        return None

    def release(self, kind: str, ts: float) -> None:
        if kind == CTRL:
            self.ctrl_is_down = False
        else:
            self.letters_down.discard(kind)


def classify_vk(vk: int) -> str:
    return _VK_KIND.get(vk, OTHER)


class HotkeyListener:
    """全局监听。忽略注入的按键事件（包括程序自己模拟的 Ctrl+V）。"""

    def __init__(self, window_ms: int, on_trigger: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic):
        self._detector = DoublePressDetector(window_ms)
        self._on_trigger = on_trigger
        self._clock = clock
        self._listener: keyboard.Listener | None = None
        self._swallowing_e = False  # 吞掉触发用的 E，直到它松开（含长按重复）

    def _filter(self, msg: int, data) -> bool:
        """在钩子线程里同步执行。返回 True 表示要吞掉这个按键事件。"""
        if data.flags & _LLKHF_INJECTED:
            return False
        kind = classify_vk(data.vkCode)

        if msg in _PRESS_MESSAGES:
            if kind == E and self._swallowing_e:
                return True
            letter = self._detector.press(kind, self._clock())
            if letter:
                threading.Thread(target=self._on_trigger, args=(letter,),
                                 name="paraphrase-worker", daemon=True).start()
                if letter == E:
                    self._swallowing_e = True
                    return True
        elif msg in _RELEASE_MESSAGES:
            self._detector.release(kind, self._clock())
            if kind == E and self._swallowing_e:
                self._swallowing_e = False
                return True
        return False

    def _win32_event_filter(self, msg, data):
        try:
            swallow = self._filter(msg, data)
        except Exception:  # 这里抛出的异常会让 pynput 停掉整个监听
            swallow = False
        if swallow:
            self._listener.suppress_event()  # 通过抛异常让钩子吞掉事件
        return False  # 不需要 pynput 再派发 on_press / on_release

    def start(self) -> None:
        self._listener = keyboard.Listener(win32_event_filter=self._win32_event_filter)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()

    def join(self, timeout: float | None = None) -> None:
        if self._listener is not None:
            self._listener.join(timeout)

    def is_alive(self) -> bool:
        return self._listener is not None and self._listener.is_alive()


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    listener = HotkeyListener(400, lambda letter: print("triggered", letter, flush=True))
    listener.start()
    print("监听中：切到其他窗口（如记事本）按 Ctrl+C C 或 Ctrl+C E 测试；在本窗口按 Ctrl+C 退出")
    try:
        while listener.is_alive():
            listener.join(0.5)
    except KeyboardInterrupt:
        listener.stop()
