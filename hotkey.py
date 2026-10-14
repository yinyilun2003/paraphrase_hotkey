"""连按热键检测：Ctrl+C 接 Ctrl+C → "c"（中文），Ctrl+C 接 Ctrl+E → "e"（英文）。

macOS 上修饰键换成 Command（Cmd+C C / Cmd+C E），下文的 Ctrl 在 macOS 上都指 Command。

DoublePressDetector 只处理抽象事件和时间戳，便于单元测试；
HotkeyListener 在系统键盘钩子的过滤回调里同步驱动它（Windows 是低级键盘钩子，macOS 是 Quartz event tap），
并在工作线程里执行 on_trigger(letter)。

按键只观察、不拦截，唯一的例外是组合里的那一下 Ctrl+E：它会被吞掉，
免得目标应用执行自己的 Ctrl+E 功能（跳到搜索框、居中等）。单独按 Ctrl+E 不受影响。

手动测试：python hotkey.py  —— 在其他窗口按 Ctrl+C C / Ctrl+C E 会打印 triggered c / e；在本窗口按 Ctrl+C 退出。
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable

from pynput import keyboard

IS_MAC = sys.platform == "darwin"
MOD_NAME = "Cmd" if IS_MAC else "Ctrl"  # 组合里的修饰键，用于提示文字

CTRL = "ctrl"  # 组合里的修饰键：Windows 是 Ctrl，macOS 是 Command
C = "c"
E = "e"
OTHER = "other"

# Windows（低级键盘钩子）
_VK_KIND = {0x11: CTRL, 0xA2: CTRL, 0xA3: CTRL, 0x43: C, 0x45: E}
_PRESS_MESSAGES = (0x0100, 0x0104)    # WM_KEYDOWN, WM_SYSKEYDOWN
_RELEASE_MESSAGES = (0x0101, 0x0105)  # WM_KEYUP, WM_SYSKEYUP
_LLKHF_INJECTED = 0x10 | 0x02         # LLKHF_INJECTED | LLKHF_LOWER_IL_INJECTED

# macOS（Quartz）。按物理键位识别，不受键盘布局和输入法影响
_MAC_KEYCODE_KIND = {0x08: C, 0x0E: E}            # kVK_ANSI_C, kVK_ANSI_E
_MAC_KEY_DOWN, _MAC_KEY_UP = 10, 11               # kCGEventKeyDown, kCGEventKeyUp
_MAC_FLAG_CMD = 0x100000                          # kCGEventFlagMaskCommand
_MAC_FLAG_OTHER_MODS = 0x20000 | 0x40000 | 0x80000  # Shift | Control | Option
_MAC_TAP_DISABLED = (0xFFFFFFFE, 0xFFFFFFFF)      # kCGEventTapDisabledByTimeout / ByUserInput

if IS_MAC:
    import HIServices
    import Quartz

    class _MacListener(keyboard.Listener):
        """记下 pynput 创建的 event tap，被系统禁用后可以重新启用。"""

        tap = None

        def _create_event_tap(self):
            self.tap = super()._create_event_tap()
            return self.tap


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


def mac_accessibility_trusted(prompt: bool = True) -> bool:
    """macOS：当前进程（实际是启动它的终端）是否有“辅助功能”权限。prompt=True 时没有权限会弹系统授权提示。"""
    return bool(HIServices.AXIsProcessTrustedWithOptions(
        {HIServices.kAXTrustedCheckOptionPrompt: prompt}))


class HotkeyListener:
    """全局监听。忽略注入的按键事件（包括程序自己模拟的 Ctrl+V）。"""

    def __init__(self, window_ms: int, on_trigger: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic):
        self._detector = DoublePressDetector(window_ms)
        self._on_trigger = on_trigger
        self._clock = clock
        self._listener: keyboard.Listener | None = None
        self._swallowing_e = False  # 吞掉触发用的 E，直到它松开（含长按重复）

    def _key(self, is_press: bool, kind: str, ts: float) -> bool:
        """处理一次真实按键（两个平台共用）。返回 True 表示要吞掉这个按键事件。"""
        if is_press:
            if kind == E and self._swallowing_e:
                return True
            letter = self._detector.press(kind, ts)
            if letter:
                threading.Thread(target=self._on_trigger, args=(letter,),
                                 name="paraphrase-worker", daemon=True).start()
                if letter == E:
                    self._swallowing_e = True
                    return True
        else:
            self._detector.release(kind, ts)
            if kind == E and self._swallowing_e:
                self._swallowing_e = False
                return True
        return False

    def _filter(self, msg: int, data) -> bool:
        """Windows：在钩子线程里同步执行。返回 True 表示要吞掉这个按键事件。"""
        if data.flags & _LLKHF_INJECTED:
            return False
        kind = classify_vk(data.vkCode)
        if msg in _PRESS_MESSAGES:
            return self._key(True, kind, self._clock())
        if msg in _RELEASE_MESSAGES:
            return self._key(False, kind, self._clock())
        return False

    def _mac_filter(self, event_type: int, keycode: int, flags: int, injected: bool) -> bool:
        """macOS：在 event tap 线程里同步执行。返回 True 表示要吞掉这个按键事件。

        Command 不单独当按键处理，而是每个事件都从 flags 里读它当前是否按着。
        """
        if injected:
            return False
        ts = self._clock()
        cmd_down = bool(flags & _MAC_FLAG_CMD)
        if cmd_down != self._detector.ctrl_is_down:
            self._key(cmd_down, CTRL, ts)

        if event_type == _MAC_KEY_DOWN:
            # Cmd+Shift+C 之类不算 Cmd+C（Windows 上按 Shift 本身就会打断连按）
            kind = OTHER if flags & _MAC_FLAG_OTHER_MODS else _MAC_KEYCODE_KIND.get(keycode, OTHER)
            return self._key(True, kind, ts)
        if event_type == _MAC_KEY_UP:
            return self._key(False, _MAC_KEYCODE_KIND.get(keycode, OTHER), ts)
        return False  # flagsChanged 等

    def _win32_event_filter(self, msg, data):
        try:
            swallow = self._filter(msg, data)
        except Exception:  # 这里抛出的异常会让 pynput 停掉整个监听
            swallow = False
        if swallow:
            self._listener.suppress_event()  # 通过抛异常让钩子吞掉事件
        return False  # 不需要 pynput 再派发 on_press / on_release

    def _darwin_intercept(self, event_type, event):
        """返回 None 表示吞掉事件，返回 event 表示照常传给应用。"""
        try:
            if event_type in _MAC_TAP_DISABLED:
                # 回调超时等原因被系统禁用后不会自己恢复，热键会静默失效
                Quartz.CGEventTapEnable(self._listener.tap, True)
                return event
            swallow = self._mac_filter(
                event_type,
                Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode),
                Quartz.CGEventGetFlags(event),
                # 与 pynput 的判断一致：注入的事件带有发送进程的 PID
                Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUnixProcessID) != 0,
            )
        except Exception:  # 这里抛出的异常会让 pynput 停掉整个监听
            swallow = False
        return None if swallow else event

    def start(self) -> None:
        if IS_MAC:
            self._listener = _MacListener(darwin_intercept=self._darwin_intercept)
        else:
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
    sys.stdout.reconfigure(encoding="utf-8")
    if IS_MAC and not mac_accessibility_trusted():
        print("没有“辅助功能”权限：请在 系统设置 → 隐私与安全性 → 辅助功能 里允许当前终端，然后重启终端再运行")
        sys.exit(2)
    listener = HotkeyListener(400, lambda letter: print("triggered", letter, flush=True))
    listener.start()
    print(f"监听中：切到其他窗口（如{'备忘录' if IS_MAC else '记事本'}）按 {MOD_NAME}+C C 或 {MOD_NAME}+C E 测试；"
          "在本窗口按 Ctrl+C 退出")
    try:
        while listener.is_alive():
            listener.join(0.5)
    except KeyboardInterrupt:
        listener.stop()
