import threading
from types import SimpleNamespace

from hotkey import C, CTRL, E, OTHER, DoublePressDetector, HotkeyListener, classify_vk


class Seq:
    """用假时间戳驱动检测器，记录触发次数。"""

    def __init__(self, window_ms=400):
        self.d = DoublePressDetector(window_ms)
        self.triggers = []

    def down(self, kind, t):
        letter = self.d.press(kind, t)
        if letter:
            self.triggers.append(t if letter == C else (letter, t))

    def up(self, kind, t):
        self.d.release(kind, t)

    def tap_c(self, t, hold=0.03):
        self.tap(C, t, hold)

    def tap(self, letter, t, hold=0.03):
        self.down(letter, t)
        self.up(letter, t + hold)


def test_double_press_triggers():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap_c(0.10)
    s.tap_c(0.30)
    assert s.triggers == [0.30]


def test_exactly_at_window_triggers():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap_c(1.0)
    s.tap_c(1.4)
    assert s.triggers == [1.4]


def test_too_slow_does_not_trigger():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap_c(0.10)
    s.tap_c(0.51)
    assert s.triggers == []


def test_slow_then_fast_triggers_on_last_pair():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap_c(0.10)
    s.tap_c(0.60)  # 超时，作为新的第一次
    s.tap_c(0.80)
    assert s.triggers == [0.80]


def test_long_press_does_not_trigger():
    s = Seq()
    s.down(CTRL, 0.0)
    s.down(C, 0.10)
    for i in range(1, 20):  # 系统自动重复
        s.down(C, 0.10 + i * 0.03)
    s.up(C, 0.80)
    assert s.triggers == []


def test_triple_press_triggers_once():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap_c(0.10)
    s.tap_c(0.25)
    s.tap_c(0.40)
    assert s.triggers == [0.25]


def test_four_presses_trigger_twice():
    s = Seq()
    s.down(CTRL, 0.0)
    for t in (0.10, 0.25, 0.40, 0.55):
        s.tap_c(t)
    assert s.triggers == [0.25, 0.55]


def test_c_without_ctrl_does_not_count():
    s = Seq()
    s.tap_c(0.10)
    s.down(CTRL, 0.15)
    s.tap_c(0.20)
    assert s.triggers == []


def test_ctrl_released_between_presses_still_triggers():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap_c(0.05)
    s.up(CTRL, 0.10)
    s.down(CTRL, 0.15)
    s.tap_c(0.20)
    assert s.triggers == [0.20]


def test_ctrl_released_then_plain_c_does_not_trigger():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap_c(0.05)
    s.up(CTRL, 0.10)
    s.tap_c(0.20)
    assert s.triggers == []


def test_other_key_between_resets():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap_c(0.10)
    s.down(OTHER, 0.15)
    s.up(OTHER, 0.17)
    s.tap_c(0.25)
    assert s.triggers == []


# ---------------------------------------------------------------------------
# Ctrl+C 接 Ctrl+E → "e"
# ---------------------------------------------------------------------------

def test_c_then_e_triggers_e():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap(C, 0.10)
    s.tap(E, 0.30)
    assert s.triggers == [(E, 0.30)]


def test_c_then_e_too_slow_does_not_trigger():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap(C, 0.10)
    s.tap(E, 0.51)
    assert s.triggers == []


def test_double_e_does_not_trigger():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap(E, 0.10)
    s.tap(E, 0.20)
    assert s.triggers == []


def test_e_then_c_does_not_trigger():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap(E, 0.10)
    s.tap(C, 0.20)
    assert s.triggers == []


def test_c_e_e_triggers_once():
    s = Seq()
    s.down(CTRL, 0.0)
    for letter, t in ((C, 0.10), (E, 0.20), (E, 0.30)):
        s.tap(letter, t)
    assert s.triggers == [(E, 0.20)]


def test_c_c_e_triggers_c_only():
    s = Seq()
    s.down(CTRL, 0.0)
    for letter, t in ((C, 0.10), (C, 0.20), (E, 0.30)):
        s.tap(letter, t)
    assert s.triggers == [0.20]


def test_c_then_e_without_ctrl_does_not_trigger():
    s = Seq()
    s.down(CTRL, 0.0)
    s.tap(C, 0.10)
    s.up(CTRL, 0.15)
    s.tap(E, 0.20)
    assert s.triggers == []


def test_classify_vk():
    assert classify_vk(0x11) == CTRL
    assert classify_vk(0xA2) == CTRL
    assert classify_vk(0xA3) == CTRL
    assert classify_vk(0x43) == C
    assert classify_vk(0x45) == E
    assert classify_vk(0x56) == OTHER  # V
    assert classify_vk(0xA0) == OTHER  # Shift


# ---------------------------------------------------------------------------
# HotkeyListener._filter：用假的 KBDLLHOOKSTRUCT 驱动
# ---------------------------------------------------------------------------

DOWN, UP = 0x0100, 0x0101
VK_CTRL, VK_C, VK_E = 0xA2, 0x43, 0x45


def ev(vk, injected=False):
    return SimpleNamespace(vkCode=vk, flags=0x10 if injected else 0)


class Recorder:
    def __init__(self):
        self.letters = []
        self.threads = []
        self.fired = threading.Event()

    def __call__(self, letter):
        self.letters.append(letter)
        self.threads.append(threading.current_thread())
        self.fired.set()


def make_listener():
    rec = Recorder()
    t = iter(i * 0.05 for i in range(1000))
    return HotkeyListener(400, rec, clock=lambda: next(t)), rec


def test_filter_c_c_triggers_in_worker_thread_and_swallows_nothing():
    lst, rec = make_listener()
    swallowed = [lst._filter(m, ev(vk)) for m, vk in
                 [(DOWN, VK_CTRL), (DOWN, VK_C), (UP, VK_C), (DOWN, VK_C), (UP, VK_C), (UP, VK_CTRL)]]
    assert rec.fired.wait(1.0)
    assert rec.letters == [C]
    assert rec.threads[0] is not threading.main_thread()
    assert not any(swallowed)


def test_filter_c_e_swallows_only_that_e_including_repeats():
    lst, rec = make_listener()
    assert not lst._filter(DOWN, ev(VK_CTRL))
    assert not lst._filter(DOWN, ev(VK_C))
    assert not lst._filter(UP, ev(VK_C))
    assert lst._filter(DOWN, ev(VK_E))       # 触发键被吞掉
    assert lst._filter(DOWN, ev(VK_E))       # 长按重复也吞掉
    assert lst._filter(UP, ev(VK_E))         # 对应的松开也吞掉
    assert rec.fired.wait(1.0)
    assert rec.letters == [E]
    # 之后单独的 Ctrl+E 照常传给应用
    assert not lst._filter(DOWN, ev(VK_E))
    assert not lst._filter(UP, ev(VK_E))


def test_filter_plain_ctrl_e_not_swallowed():
    lst, rec = make_listener()
    assert not lst._filter(DOWN, ev(VK_CTRL))
    assert not lst._filter(DOWN, ev(VK_E))
    assert not lst._filter(UP, ev(VK_E))
    assert not rec.fired.is_set()


def test_filter_ignores_injected_events():
    lst, rec = make_listener()
    for m, vk in [(DOWN, VK_CTRL), (DOWN, VK_C), (UP, VK_C), (DOWN, VK_C), (DOWN, VK_E)]:
        assert not lst._filter(m, ev(vk, injected=True))
    assert not rec.fired.is_set()


def test_event_filter_never_lets_pynput_dispatch_and_survives_errors():
    lst, rec = make_listener()
    lst._listener = SimpleNamespace(suppress_event=lambda: None)
    assert lst._win32_event_filter(DOWN, ev(VK_CTRL)) is False
    assert lst._win32_event_filter(DOWN, object()) is False  # 坏数据不抛异常
