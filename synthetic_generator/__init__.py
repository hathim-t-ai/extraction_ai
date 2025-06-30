from importlib.metadata import PackageNotFoundError, version as _pkg_version

__all__ = [
    "generate_statements",
]

try:
    __version__ = _pkg_version(__name__.split(".")[0])
except PackageNotFoundError:
    __version__ = "0.0.0"

from .cli import generate_statements  # noqa: E402 