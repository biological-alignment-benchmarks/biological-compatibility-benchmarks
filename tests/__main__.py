import os

import pytest

from aintelope.config.config_utils import register_resolvers

if __name__ == "__main__" and os.name == "nt":  # detect debugging
    register_resolvers()  # needs to be called only once, needed for test_trainer.py
    pytest.main(["tests/"])  # run tests only in this folder and its subfolders
