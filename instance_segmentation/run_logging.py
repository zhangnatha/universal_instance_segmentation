from __future__ import annotations

import atexit
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TextIO


class _PrefixedTee:
    """将 stdout/stderr 重定向到终端和日志文件，并在每行开头添加时间戳。"""

    def __init__(
        self,
        terminal: TextIO,
        log_file: TextIO,
        module: str,
        lock: threading.Lock,
        terminal_output: bool = True,
    ):
        self.terminal = terminal
        self.log_file = log_file
        self.module = module
        self.lock = lock
        self.terminal_output = terminal_output
        self.at_line_start = True

    def _format_prefix(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.module:
            return f"[{ts}] [{self.module}] "
        return f"[{ts}] "

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self.lock:
            for character in text:
                if self.at_line_start and character not in "\r\n":
                    prefix = self._format_prefix()
                    if self.terminal_output:
                        self.terminal.write(prefix)
                    self.log_file.write(prefix)
                    self.at_line_start = False
                if self.terminal_output:
                    self.terminal.write(character)
                self.log_file.write(character)
                if character == "\n":
                    self.at_line_start = True
                elif character == "\r":
                    pass
            if self.terminal_output:
                self.terminal.flush()
            self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            if self.terminal_output:
                self.terminal.flush()
            self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", "utf-8")


_CURRENT_LOG_PATH: Path | None = None
_LOGGING_LOCK = threading.Lock()


def setup_run_logging(
    module: str,
    *,
    terminal_output: bool = True,
    log_dir: str | Path | None = None,
) -> Path:
    """设置带有时间戳的控制台日志记录，并将完整输出持久化到按日期划分的日志目录（logs/YYYY-MM-DD/）。"""
    global _CURRENT_LOG_PATH

    with _LOGGING_LOCK:
        if _CURRENT_LOG_PATH is not None:
            return _CURRENT_LOG_PATH

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")

        if log_dir is None:
            base_dir = Path("logs") / date_str
        else:
            base_dir = Path(log_dir) / date_str

        base_dir.mkdir(parents=True, exist_ok=True)
        sanitized_module = module.strip().replace(" ", "_").replace("/", "_").lower()
        log_path = base_dir / f"{sanitized_module}_{time_str}.log"
        log_file = log_path.open("a", encoding="utf-8", buffering=1)

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        lock = threading.Lock()

        sys.stdout = _PrefixedTee(
            original_stdout, log_file, module, lock, terminal_output
        )
        sys.stderr = _PrefixedTee(
            original_stderr, log_file, module, lock, terminal_output
        )

        def close_log() -> None:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            try:
                log_file.close()
            except Exception:
                pass

        atexit.register(close_log)
        _CURRENT_LOG_PATH = log_path
        print(f"Logging initialized. Date: {date_str}, Log file: {log_path}")
        return log_path
