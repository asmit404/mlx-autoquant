from __future__ import annotations


class AutoQuantError(RuntimeError):
    """Expected operational failure with a stable machine-readable category."""

    code = "error"

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class MetadataError(AutoQuantError):
    code = "metadata"


class AuthenticationError(AutoQuantError):
    code = "authentication"


class NetworkError(AutoQuantError):
    code = "network"


class UnsupportedModelError(AutoQuantError):
    code = "unsupported_model"


class InsufficientMemoryError(AutoQuantError):
    code = "insufficient_memory"


class InsufficientDiskError(AutoQuantError):
    code = "insufficient_disk"


class ConversionError(AutoQuantError):
    code = "conversion"


class VerificationError(AutoQuantError):
    code = "verification"


class CancellationError(AutoQuantError):
    code = "cancellation"
