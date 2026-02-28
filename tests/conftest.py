"""
Patch multiprocessing.Queue so that every instance created during the test
session automatically has cancel_join_thread() called.

On Windows, multiprocessing.Queue starts a background non-daemon feeder
thread the first time put() is called.  Without cancel_join_thread(), Python
waits for that thread during interpreter shutdown, which causes pytest to hang
indefinitely after all tests have finished.  cancel_join_thread() tells the
Queue not to wait for the feeder thread on close(), which is always correct
for tests because they never span multiple OS processes.
"""
from __future__ import annotations

import multiprocessing.queues

import pytest

_original_queue_init = multiprocessing.queues.Queue.__init__


def _queue_init_no_hang(self, *args, **kwargs):
    _original_queue_init(self, *args, **kwargs)
    self.cancel_join_thread()


@pytest.fixture(autouse=True, scope="session")
def _patch_mp_queues_for_clean_exit():
    multiprocessing.queues.Queue.__init__ = _queue_init_no_hang
    yield
    multiprocessing.queues.Queue.__init__ = _original_queue_init
