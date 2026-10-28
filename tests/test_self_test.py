"""Run the installation self-test as part of the suite."""

import pytest


@pytest.mark.slow
def test_self_test_passes():
    from falcocv.self_test import main

    assert main() == 0
