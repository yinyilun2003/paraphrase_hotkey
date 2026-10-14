"""入口：加载配置、启动监听。Ctrl+C C / Ctrl+C E 把选中文字原地替换成中文 / 英文改写结果。"""

from __future__ import annotations

import logging
import statistics
import sys
import threading
import time
import winsound

import clipboard_io
from config import Config, ConfigError, load_config
from hotkey import HotkeyListener
from paraphrase import ParaphraseError, paraphrase

log = logging.getLogger("paraphrase-hotkey")

# Ctrl+C 接 Ctrl+C → 中文，Ctrl+C 接 Ctrl+E → 英文，与输入语言无关
LANG_BY_LETTER = {"c": "zh", "e": "en"}


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._busy = threading.Lock()
        self.durations_ms: list[float] = []  # 成功请求的端到端耗时，退出时汇总

    def on_trigger(self, letter: str = "c") -> None:
        if not self._busy.acquire(blocking=False):
            log.info("result=ignored reason=上一次还在处理")
            return
        try:
            self._handle(letter)
        except Exception as e:  # 兜底：任何意外都不能让监听线程崩掉
            log.error("result=failed reason=意外错误（%s）", type(e).__name__)
            _beep()
        finally:
            self._busy.release()

    def _handle(self, letter: str) -> None:
        t0 = time.monotonic()
        lang = LANG_BY_LETTER[letter]
        hwnd = clipboard_io.foreground_window()
        # 两种组合的第一下都是用户自己的 Ctrl+C，选区已经在剪贴板里
        clipboard_io.wait_settle()
        original = clipboard_io.read_text()

        if original is None or not original.strip():
            log.info("result=skipped reason=剪贴板为空或非文本")
            return
        in_chars = len(original)
        if in_chars > self.cfg.max_input_chars:
            log.warning("in=%d result=skipped reason=文本过长，已跳过（上限 %d）", in_chars, self.cfg.max_input_chars)
            _beep()
            return
        if self.cfg.log_text:
            log.info("input=%r", original)

        try:
            result = paraphrase(original, lang, cfg=self.cfg)
        except ParaphraseError as e:
            self._fail(t0, in_chars, str(e))
            return

        if self.cfg.log_text:
            log.info("output=%r", result)
        if result == original.strip():
            log.info("in=%d out=%d ms=%d result=unchanged reason=无需改写，未粘贴",
                     in_chars, len(result), _ms(t0))
            return

        if not clipboard_io.wait_ctrl_released():
            self._fail(t0, in_chars, "Ctrl 一直未松开，未粘贴")
            return
        if clipboard_io.foreground_window() != hwnd:
            self._fail(t0, in_chars, "窗口已切换，未粘贴")
            return

        clipboard_io.write_text(result)
        try:
            clipboard_io.paste()
        finally:
            clipboard_io.restore(original)

        elapsed = _ms(t0)
        self.durations_ms.append(elapsed)
        log.info("lang=%s in=%d out=%d ms=%d result=ok", lang, in_chars, len(result), elapsed)

    def _fail(self, t0: float, in_chars: int, reason: str) -> None:
        log.warning("in=%d ms=%d result=failed reason=%s", in_chars, _ms(t0), reason)
        _beep()

    def summary(self) -> str:
        d = self.durations_ms
        if not d:
            return "本次运行没有成功的改写"
        p50 = statistics.median(d)
        p95 = statistics.quantiles(d, n=20, method="inclusive")[-1] if len(d) >= 2 else d[0]
        return f"成功 {len(d)} 次，端到端耗时 p50={p50:.0f} ms，p95={p95:.0f} ms"


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _beep() -> None:
    try:
        winsound.MessageBeep()
    except Exception:
        pass


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    try:
        cfg = load_config()
    except ConfigError as e:
        log.error("配置错误：%s", e)
        return 2

    app = App(cfg)
    listener = HotkeyListener(cfg.double_press_window_ms, app.on_trigger)
    listener.start()
    log.info("已启动（模型 %s）。在任意文本框 Ctrl+A 后快速连按两次 Ctrl+C 改写成中文，"
             "按 Ctrl+C 再按 Ctrl+E 改写成英文；"
             "在本窗口按 Ctrl+C 退出。", cfg.model)
    try:
        while listener.is_alive():
            listener.join(0.5)
        log.error("键盘监听已意外停止")
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        log.info(app.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
