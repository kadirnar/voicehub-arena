import sys
import subprocess
import os
import time
import pytest
from voicehub_arena.cli import isolated


def test_timeout_is_recorded_and_child_stopped(tmp_path):
    result=isolated([sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'worker.log',.1)
    assert result==124


def test_worker_exit_status_preserved(tmp_path):
    assert isolated([sys.executable,'-c','raise SystemExit(7)'],tmp_path/'worker.log',3)==7


def test_stopping_controller_reaps_its_worker(tmp_path):
    log=tmp_path/'nested.log'
    child="import os,time; print(os.getpid(),flush=True); time.sleep(30)"
    code=f"from voicehub_arena.cli import isolated; isolated([{sys.executable!r},'-c',{child!r}],{str(log)!r},60)"
    controller=subprocess.Popen([sys.executable,'-c',code],stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while not (log.exists() and log.read_text().strip()):
            if time.monotonic()>deadline:
                pytest.fail('Child did not start')
            time.sleep(.02)
        pid=int(log.read_text().strip())
        controller.terminate()
        controller.wait(timeout=5)
        with pytest.raises(ProcessLookupError):
            os.kill(pid,0)
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()
