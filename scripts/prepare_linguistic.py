"""Prepare audited English frontends on CPU; save tensors and source provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("NLTK_DATA", str(Path(__file__).resolve().parents[1]/'.cache/nltk'))
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

root = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["melotts", "gptsovits"])
parser.add_argument('--consumer', choices=['melotts', 'gptsovits', 'openvoice'])
parser.add_argument('--dataset', default='datasets/english.jsonl')
parser.add_argument('--output')
parser.add_argument('--overrides', default='configs/models.json')
parser.add_argument('--input-text-transform', choices=['identity','librispeech_lowercase_v1'], default='identity')
args = parser.parse_args()
consumer = args.consumer or args.model
if consumer != args.model and not (consumer == 'openvoice' and args.model == 'melotts'):
    parser.error('OpenVoice can reuse only the MeloTTS frontend')
dataset_path = root/args.dataset
rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line]
from voicehub_arena.inputs import prepare_benchmark_text
rows = [{**row, 'text':prepare_benchmark_text(row['text'], args.input_text_transform)} for row in rows]
destination = root/args.output if args.output else root/"datasets/prepared"/args.model
destination.mkdir(parents=True, exist_ok=True)
manifest = {"model": args.model, "language": "en", "entries": {}, "device": "cpu",
            'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            'input_text_transform': args.input_text_transform,
            "timing_scope": "synthesis with offline linguistic inputs; preparation measured separately"}

def save(row, started, **arrays):
    key = hashlib.sha256(row["text"].encode()).hexdigest()
    file = destination/f"{key}.npz"
    np.savez_compressed(file, **{k: np.asarray(v) for k,v in arrays.items()})
    manifest["entries"][key] = {"id": row["id"], "file": file.name,
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "preparation_s": time.perf_counter()-started}
    print(args.model, row["id"], "prepared", flush=True)

if args.model == "melotts":
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from voicehub.architectures.melotts.metadata import MELOTTS_RELEASES
    from voicehub.models.melotts.source.melo.text import english, cleaned_text_to_sequence
    repo, revision, _, _ = MELOTTS_RELEASES["EN"]
    data = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    bert_repo = "google-bert/bert-base-uncased"
    bert_revision = HfApi().model_info(bert_repo).sha
    tokenizer = AutoTokenizer.from_pretrained(bert_repo, revision=bert_revision)
    bert_model = None if data["data"].get("disable_bert", False) else AutoModelForMaskedLM.from_pretrained(bert_repo, revision=bert_revision).eval()
    symbol_ids = {symbol:i for i,symbol in enumerate(data["symbols"])}
    manifest.update(checkpoint=repo, revision=revision, bert_repo=bert_repo, bert_revision=bert_revision,
        source="VoiceHub vendored MeloTTS English G2P and released BERT hidden-state recipe",
        zero_channels="The unused 1024-channel Chinese BERT branch is zero by the upstream English protocol.")
    for row in rows:
        started = time.perf_counter()
        normalized = english.text_normalize(row["text"])
        phones, tones, word2ph = english.g2p(normalized, tokenized=tokenizer.tokenize(normalized))
        phones, tones, languages = cleaned_text_to_sequence(phones, tones, "EN", symbol_to_id=symbol_ids)
        if data["data"].get("add_blank"):
            def intersperse(values):
                result = [0]*(len(values)*2+1); result[1::2] = values; return result
            phones, tones, languages = map(intersperse, (phones, tones, languages))
            word2ph = [count*2 for count in word2ph]; word2ph[0] += 1
        if bert_model is None:
            features = torch.zeros(768, len(phones))
        else:
            tokens = tokenizer(normalized, return_tensors="pt")
            if tokens["input_ids"].shape[-1] != len(word2ph):
                raise ValueError("English BERT tokens and phoneme alignment disagree")
            with torch.no_grad():
                hidden = bert_model(**tokens, output_hidden_states=True).hidden_states[-3][0]
            features = torch.cat([hidden[i].repeat(n,1) for i,n in enumerate(word2ph)], dim=0).T
        if features.shape != (768, len(phones)):
            raise ValueError("English BERT feature shape mismatch")
        save(row, started, input_ids=np.array(phones,dtype=np.int64), tone_ids=np.array(tones,dtype=np.int64),
            language_ids=np.array(languages,dtype=np.int64), bert_features=np.zeros((1024,len(phones)),np.float32),
            ja_bert_features=features.numpy())
else:
    import librosa
    import soundfile as sf
    import torchaudio
    from transformers import HubertModel
    from voicehub.models.gptsovits.source.GPT_SoVITS.text.cleaner import clean_text
    from voicehub.models.gptsovits.source.GPT_SoVITS.text import cleaned_text_to_sequence
    from voicehub.models.gptsovits.source.GPT_SoVITS.module.mel_processing import spectrogram_torch
    # This exact official example is 3-10 s as required by upstream prompting.
    reference = root/"datasets/reference/emily.wav"
    transcript = (root/"datasets/reference/emily.txt").read_text().strip()
    wave16, _ = librosa.load(reference, sr=16000)
    if not 48000 <= len(wave16) <= 160000:
        raise ValueError("GPT-SoVITS requires an authorized 3-10 second reference with exact transcript")
    repo = "TencentGameMate/chinese-hubert-base"
    revision = HfApi().model_info(repo).sha
    hubert = HubertModel.from_pretrained(repo, revision=revision).eval()
    # Upstream appends 0.3 * 32000 zero samples to its 16 kHz reference.
    padded = torch.cat([torch.from_numpy(wave16), torch.zeros(int(32000*.3))])
    with torch.no_grad():
        ssl = hubert(padded.unsqueeze(0)).last_hidden_state.transpose(1,2).numpy()
    del hubert
    raw, sr = sf.read(reference, dtype="float32", always_2d=True)
    waveform = torchaudio.functional.resample(torch.from_numpy(raw.mean(1)).unsqueeze(0), sr, 32000)
    peak = waveform.abs().max()
    if peak > 1:
        waveform /= min(2,float(peak))
    spec = spectrogram_torch(waveform, 2048, 32000, 640, 2048, center=False).numpy()
    np.savez_compressed(destination/"reference.npz", ssl_features=ssl, reference_spectrogram=spec)
    ref_phones, _, _ = clean_text(transcript, "en", "v2")
    ref_ids = cleaned_text_to_sequence(ref_phones, "v2")
    manifest.update(hubert_repo=repo, hubert_revision=revision, source="VoiceHub vendored GPT-SoVITS v2 English G2P and reference feature recipe",
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        zero_channels="1024-dimensional BERT is zero for English in upstream get_bert_inf; it is only computed for Chinese.")
    for row in rows:
        started = time.perf_counter()
        phones, _, _ = clean_text(row["text"], "en", "v2")
        target_ids = cleaned_text_to_sequence(phones, "v2")
        ids = ref_ids+target_ids
        save(row, started, s1_phoneme_ids=np.array(ids,dtype=np.int64), s2_phoneme_ids=np.array(target_ids,dtype=np.int64),
            s1_bert_features=np.zeros((1,1024,len(ids)),np.float32))

(destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
path = root/args.overrides
config = json.loads(path.read_text())
override = config.setdefault(consumer,{})
override["prepared_inputs"] = str(destination.relative_to(root))
if consumer != 'openvoice':
    override["timing_scope"] = manifest["timing_scope"]
    override.setdefault("config", {})["trust_pickle_checkpoint"] = True
    if args.model == "gptsovits":
        override["config"]["revision"] = "336b2ec4e8d4ac74740798dd40af44e74659ecaf"
    else:
        override["config"]["revision"] = revision
path.write_text(json.dumps(config,indent=2)+"\n")
print(args.model, "preparation complete", flush=True)
