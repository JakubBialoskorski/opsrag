import time

from opsrag.ingest import minutes_left


def test_minutes_left_extrapolates_from_elapsed_time():
    started = time.monotonic() - 59  # 10 of 40 documents took about a minute
    assert minutes_left(started, done=10, total=40) == 3


def test_minutes_left_is_zero_when_done():
    assert minutes_left(time.monotonic() - 5, done=7, total=7) == 0
