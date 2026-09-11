import sys
from voicehub_arena.cli import isolated


def test_timeout_is_recorded_and_child_stopped(tmp_path):
    result=isolated([sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'worker.log',.1)
    assert result==124


def test_worker_exit_status_preserved(tmp_path):
    assert isolated([sys.executable,'-c','raise SystemExit(7)'],tmp_path/'worker.log',3)==7
