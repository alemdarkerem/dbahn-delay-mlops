"""Smoke test proving the package installs and imports correctly."""

import dbahn_delay


def test_package_has_version() -> None:
    assert dbahn_delay.__version__ == "0.1.0"
