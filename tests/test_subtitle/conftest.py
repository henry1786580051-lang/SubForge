"""Subtitle test configuration.

Most subtitle tests exercise pure parsing, rendering, and validation code and do
not require a Qt application. Qt-dependent modules request the shared fixture
explicitly so headless test runs cannot abort before unrelated tests execute.
"""

import sys

import pytest
from PyQt5.QtCore import QCoreApplication


@pytest.fixture(scope="session")
def qapp():
    """Create the windowless event loop required by QThread signal tests."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv)
    yield app
