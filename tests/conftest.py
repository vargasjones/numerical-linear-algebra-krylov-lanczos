import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(autouse=True)
def single_threaded_blas():
    with threadpool_limits(limits=1):
        yield
