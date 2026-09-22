"""GPU 健康检查与异常诊断工具模块。"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path


_SYSLOG_TIMESTAMP = re.compile(r"^(?P<stamp>[A-Z][a-z]{2}\s+\d+\s+\d\d:\d\d:\d\d)\s")
_FATAL_XIDS = re.compile(r"Xid .*?:\s*(79|154),|GPU has fallen off the bus", re.IGNORECASE)


def _nvidia_smi_power_limits() -> list[float]:
    """在不初始化 CUDA 上下文的情况下返回当前活动的功率限制。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.limit",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Cannot query the NVIDIA GPU with nvidia-smi. Reboot after an Xid 79 "
            "failure, then verify the NVIDIA driver before starting training."
        ) from error
    try:
        return [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except ValueError as error:
        raise RuntimeError(f"Unexpected nvidia-smi power-limit output: {result.stdout!r}") from error


def require_gpu_power_limit(max_watts: float) -> None:
    """当未配置管理员设置的安全功率上限时，拒绝长时间 CUDA 运行。"""
    if max_watts <= 0:
        raise ValueError("--max-gpu-power-watts must be greater than zero")
    limits = _nvidia_smi_power_limits()
    if not limits:
        raise RuntimeError("nvidia-smi did not report any NVIDIA GPU")
    unsafe = [value for value in limits if value > max_watts + 0.01]
    if unsafe:
        watts = f"{max_watts:g}"
        raise RuntimeError(
            f"GPU power limit is {max(unsafe):g} W, above the requested safe limit "
            f"of {watts} W. This machine has recorded NVIDIA Xid 79 (GPU fallen "
            f"off the PCIe bus), which also terminates the desktop display. Run "
            f"'sudo nvidia-smi --power-limit={watts}' once after boot, then retry."
        )
    print(f"GPU safety check: active power limit {max(limits):g} W <= {max_watts:g} W")


def fatal_xid_lines(log_text: str, started_at: datetime) -> list[str]:
    """提取在当前训练过程中记录的致命 NVIDIA 事件。"""
    threshold = started_at - timedelta(seconds=5)
    matches: list[str] = []
    for line in log_text.splitlines():
        if not _FATAL_XIDS.search(line):
            continue
        stamp_match = _SYSLOG_TIMESTAMP.match(line)
        if not stamp_match:
            continue
        try:
            stamp = datetime.strptime(
                f"{started_at.year} {stamp_match.group('stamp')}", "%Y %b %d %H:%M:%S"
            )
        except ValueError:
            continue
        if stamp >= threshold:
            matches.append(line)
    return matches


def explain_cuda_failure(error: BaseException, started_at: datetime) -> str | None:
    """将异步 CUDA 错误转换为内核级别的诊断信息。"""
    message = str(error).lower()
    if not any(token in message for token in ("cuda", "cudnn", "launch failure")):
        return None
    syslog_path = os.environ.get("INSTANCE_SEG_SYSLOG")
    if not syslog_path:
        return None
    try:
        # 日志路径取决于具体运行环境；切勿将宿主机绝对路径硬编码到项目中。
        # 仅读取末尾部分以避免加载过大的系统日志文件。
        with Path(syslog_path).expanduser().open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 512_000))
            log_text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    events = fatal_xid_lines(log_text, started_at)
    if not events:
        return None
    return (
        "NVIDIA Xid 79/154 detected: the GPU disconnected from the PCIe bus; "
        "this is why the display went black. This is not a dataset or "
        "Detectron2 out-of-memory error. A reboot is required. After reboot, "
        "apply the configured power limit and resume from last_checkpoint with "
        "the same command plus --resume. Kernel event: "
        + events[-1].strip()
    )
