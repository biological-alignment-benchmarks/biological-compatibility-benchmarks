import pytest
import sys
from aintelope.config.config_utils import register_resolvers

if __name__ == "__main__" and sys.gettrace() is not None:  # detect debugging
    register_resolvers()  # needs to be called only once, needed for test_trainer.py
    pytest.main(["tests/"])  # run tests only in this folder and its subfolders
