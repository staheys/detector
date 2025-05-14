try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        # Match the name in pyproject.toml
        __version__ = version("smart-detector-person")
    except PackageNotFoundError:
        __version__ = "0.0.0-dev"
except ImportError:
    import pkg_resources
    try:
        __version__ = pkg_resources.get_distribution("smart-detector-person").version
    except pkg_resources.DistributionNotFound:
        __version__ = "0.0.0-dev"

APP_NAME = "PersonDetector"