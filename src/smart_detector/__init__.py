# src/smart_detector/__init__.py
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("smart-detector-dual-stream") # Match pyproject.toml
    except PackageNotFoundError:
        __version__ = "0.4.0-dev"
except ImportError:
    import pkg_resources
    try:
        __version__ = pkg_resources.get_distribution("smart-detector-dual-stream").version
    except pkg_resources.DistributionNotFound:
        __version__ = "0.4.0-dev"

APP_NAME = "DualStreamDetector"