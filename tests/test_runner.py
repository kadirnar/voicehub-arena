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


@pytest.mark.parametrize('resume_samples', [False, True])
def test_partial_generation_is_retried_only_when_sample_resume_requested(tmp_path, monkeypatch, resume_samples):
    import json
    from types import SimpleNamespace
    from voicehub_arena import cache, cli
    run = tmp_path / 'run'
    (run / 'vits').mkdir(parents=True)
    config = {'catalog': [{'model_type': 'vits', 'checkpoint': 'fixed/checkpoint'}],
              'resume_samples': True, 'timeout_s': 30, 'min_free_gib': 0}
    config_path = run / 'config.json'
    config_path.write_text(json.dumps(config))
    frozen = config_path.read_bytes()
    result_path = run / 'vits/result.json'
    result_path.write_text(json.dumps({'model_type': 'vits', 'status': 'partial', 'rows': [
        {'id': 'good', 'status': 'ok', 'audio_sha256': 'preserve'},
        {'id': 'retry', 'status': 'failed'}]}))
    calls = []

    def fake_worker(command, log, timeout):
        calls.append(command[3])
        # The worker must receive the original run config, whose sample-resume
        # setting makes it retain verified audio and synthesize missing rows.
        assert Path(command[4]).read_bytes() == frozen
        if command[3] == 'generate':
            payload = json.loads(result_path.read_text())
            assert payload['rows'][0]['audio_sha256'] == 'preserve'
            payload['rows'][1]['status'] = 'generated'
            payload['status'] = 'generated'
            result_path.write_text(json.dumps(payload))
        return 0

    from pathlib import Path
    monkeypatch.setattr(cli, 'isolated', fake_worker)
    monkeypatch.setattr(cli, 'report', lambda root: {'results': [json.loads(result_path.read_text())]})
    monkeypatch.setattr(cache, 'cleanup_dead_download_locks', lambda: [])
    monkeypatch.setattr(cache, 'cleanup_abandoned_downloads', lambda: {'files': 0, 'bytes': 0})
    cli.execute(SimpleNamespace(output=str(run), resume=True, resume_samples=resume_samples))
    assert ('generate' in calls) == resume_samples
    assert config_path.read_bytes() == frozen
    assert json.loads(result_path.read_text())['rows'][0]['audio_sha256'] == 'preserve'
