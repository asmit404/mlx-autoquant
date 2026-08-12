from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareProfile:
    chip: str
    memory_bytes: int
    is_apple_silicon: bool

    @property
    def memory_gib(self) -> float:
        return self.memory_bytes / 1024**3

    def to_dict(self) -> dict:
        data = asdict(self)
        data["memory_gib"] = round(self.memory_gib, 2)
        return data


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def detect_hardware() -> HardwareProfile:
    """Read only local machine information; no model is loaded."""
    machine = platform.machine().lower()
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or machine
    memory = _sysctl("hw.memsize")
    if not memory:
        raise RuntimeError("Could not determine system memory (hw.memsize).")
    return HardwareProfile(
        chip=chip,
        memory_bytes=int(memory),
        is_apple_silicon=machine in {"arm64", "aarch64"} and platform.system() == "Darwin",
    )
