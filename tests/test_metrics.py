import numpy as np
import pytest
from voicehub_arena.metrics import errors, normalize, signal_metrics, summarize


def test_corpus_wer_is_micro_averaged():
    result=errors(["one", "one two three four five six seven eight nine"],["wrong","one two three four five six seven eight nine"])
    assert result["wer"]==pytest.approx(.1)
    assert result["word_substitutions"]==1


def test_deletions_insertions_empty_and_punctuation():
    assert errors(["hello world"],[""])["wer"]==1
    assert errors(["one"],["one two three"])["wer"]==2
    assert errors(["Hello, world!"],["hello world"])["wer"]==0
    assert normalize("I’m   READY.")=="im ready"


def test_signal_duration_silence_clipping():
    x=np.r_[np.zeros(200),np.ones(400),np.zeros(400)]
    result=signal_metrics(x,1000)
    assert result["duration_s"]==1
    assert result["silence_ratio"]==pytest.approx(.6)
    assert result["clipping_ratio"]==pytest.approx(.4)
    assert result["leading_silence_s"]==pytest.approx(.2)
    assert result["trailing_silence_s"]==pytest.approx(.4)


def test_invalid_audio_rejected():
    for audio in ([],[float("nan")],[float("inf")]):
        with pytest.raises(ValueError):
            signal_metrics(audio,16000)


def test_failures_never_get_quality_scores():
    assert "wer" not in summarize([{"status":"failed"}])
    assert summarize([])["generation_failure_rate"] is None


def test_confidence_clusters_repeat_samples():
    rows=[]
    for i in range(3):
        for repeat in range(2):
            rows.append(dict(status="ok",id=str(i),category="test",reference="one two",
                 transcript="one two",latency_s=1,duration_s=2,peak_vram_mib=10,
                 clipping_ratio=0,silence_ratio=0,rms_dbfs=-20))
    result=summarize(rows)
    assert result["unique_prompts_scored"]==3
    assert result["wer_ci95"]==[0,0]
    assert result["rtf"]==.5
