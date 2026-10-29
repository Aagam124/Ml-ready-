"""Importing this package registers every built-in check."""
from app.engine.checks import quality, outliers, mlrisk  # noqa: F401
