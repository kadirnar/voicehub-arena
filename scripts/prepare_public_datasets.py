"""Freeze publisher test texts, source digests and deterministic coverage panels.

This imports data only. It never executes dataset repositories or synthesizes
baseline audio. Seed prompt metadata is retained, but this suite measures fixed
voice intelligibility, not the publisher's zero-shot speaker-similarity track.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

import requests


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def download(url, path):
    path = Path(path)
    if path.exists():
        return path
    temporary = path.with_suffix(path.suffix + '.incomplete')
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temporary.open('wb') as output:
            for chunk in response.iter_content(8 * 2**20):
                output.write(chunk)
    temporary.replace(path)
    return path


def balanced_order(rows):
    """Round-robin strata, with hash order inside each; no score-based sampling."""
    strata = defaultdict(list)
    for row in rows:
        length = len(row['text'].split())
        bucket = 'short' if length < 10 else 'medium' if length < 30 else 'long'
        strata[(row['category'], str(row.get('evolution_depth', '')), bucket)].append(row)
    for group in strata.values():
        group.sort(key=lambda row: hashlib.sha256(('42:' + row['id']).encode()).hexdigest())
    ordered = []
    for index in range(max(map(len, strata.values()))):
        for key in sorted(strata):
            if index < len(strata[key]):
                ordered.append(strata[key][index])
    return ordered


def seed_rows(archive):
    with tarfile.open(archive) as tar:
        matches = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('/en/meta.lst')]
        if len(matches) != 1:
            raise ValueError('Expected exactly one official en/meta.lst')
        data = tar.extractfile(matches[0]).read()
    rows = []
    for line in data.decode().splitlines():
        fields = line.strip().split('|')
        if len(fields) not in (4, 5):
            raise ValueError('Unexpected Seed-TTS metadata layout')
        identifier, prompt_text, prompt_audio, text = fields[:4]
        rows.append(dict(id='seed_' + identifier, source_id=identifier, text=text,
                         reference=text, category='read_speech', language='en',
                         prompt_text=prompt_text, publisher_prompt_audio=prompt_audio,
                         publisher_target_audio=fields[4] if len(fields) == 5 else None))
    return rows, hashlib.sha256(data).hexdigest()


def libri_rows(archive, corpus, split):
    rows = []
    texts = defaultdict(dict)
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile():
                continue
            if corpus == 'libritts' and member.name.endswith(('.original.txt', '.normalized.txt')):
                stem, variant, _ = Path(member.name).name.rsplit('.', 2)
                texts[stem][variant] = tar.extractfile(member).read().decode().strip()
            elif corpus == 'librispeech' and member.name.endswith('.trans.txt'):
                for line in tar.extractfile(member).read().decode().splitlines():
                    identifier, text = line.split(' ', 1)
                    texts[identifier] = {'original': text, 'normalized': text}
    for identifier, variants in sorted(texts.items()):
        if set(variants) != {'original', 'normalized'}:
            raise ValueError('Missing original or normalized transcript: ' + identifier)
        rows.append(dict(id=corpus + '_' + identifier, source_id=identifier,
                         text=variants['original'], reference=variants['normalized'],
                         category='read_speech', language='en',
                         speaker_id=identifier.replace('_', '-').split('-')[0], split=split))
    return rows


def write_dataset(destination, key, rows, provenance, panel_size, shard_size):
    import re
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate dataset: ' + key)
    for row in rows:
        if not re.fullmatch('[A-Za-z0-9_-]+', row['id']) or not row['text'].strip() or not row['reference'].strip():
            raise ValueError('Invalid dataset row: ' + row['id'])
        row['dataset_id'] = key
        row['track'] = 'fixed_voice_intelligibility'
    rows = balanced_order(rows)
    root = destination / key
    root.mkdir(parents=True, exist_ok=True)
    def emit(path, items):
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in items))
        return {'path': str(path.relative_to(destination.parent.parent)),
                'sha256': digest(path), 'samples': len(items)}
    full = emit(root / 'full.jsonl', rows)
    shards = []
    # The first panel is a prefix of the full deterministic order. Expansion
    # evaluates only the remainder, so it cannot count early samples twice.
    boundaries = [0, min(32, len(rows)), min(panel_size, len(rows))]
    boundaries.extend(range(boundaries[-1] + shard_size, len(rows), shard_size))
    boundaries.append(len(rows))
    boundaries = sorted(set(boundaries))
    for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        shards.append({**emit(root / f'shard-{i:03d}.jsonl', rows[start:end]),
                       'index': i, 'phase': 'pilot' if i == 0 else 'panel' if i == 1 else 'expansion'})
    manifest = dict(id=key, language='en', total_samples=len(rows), full=full, shards=shards,
                    categories=dict(Counter(r['category'] for r in rows)),
                    track='fixed_voice_intelligibility', provenance=provenance,
                    selection='Full split, seed-42 SHA256 order within category/depth/length strata; round-robin strata',
                    repeats=1, synthesis_seed=42)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(key, len(rows), 'texts,', len(shards), 'shards', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', default='datasets/public')
    parser.add_argument('--panel-size', type=int, default=256)
    parser.add_argument('--shard-size', type=int, default=256)
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    cache = Path(args.cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    manifests = []

    repo = 'bosonai/EmergentTTS-Eval'
    revision = 'a7406fa315a1df2a5ffcc1a782404648bf84fbd3'
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True, token=False)
    rows, sources = [], []
    for file in sorted(info.siblings, key=lambda f: f.rfilename):
        if not file.rfilename.endswith('.parquet'):
            continue
        path = hf_hub_download(repo, file.rfilename, repo_type='dataset', revision=revision,
                               cache_dir=cache / 'hub', token=False)
        sha = digest(path)
        if file.lfs and sha != file.lfs.sha256:
            raise ValueError('Publisher Parquet digest mismatch')
        sources.append({'filename': file.rfilename, 'sha256': sha})
        table = pq.read_table(path, columns=['category', 'text_to_synthesize', 'evolution_depth', 'language'])
        for row in table.to_pylist():
            source_id = len(rows)
            if row['language'] != 'en':
                raise ValueError('Unexpected non-English row')
            rows.append(dict(id=f'emergent_{source_id:04d}', source_id=source_id, text=row['text_to_synthesize'],
                             reference=row['text_to_synthesize'], category=row['category'],
                             evolution_depth=row['evolution_depth'], language='en'))
    if len(rows) != 1645:
        raise ValueError('Pinned EmergentTTS split must contain 1645 rows')
    manifests.append(write_dataset(destination, 'emergenttts', rows, dict(repo=repo, revision=revision,
        split='train (publisher evaluation split)', source_url='https://huggingface.co/datasets/' + repo,
        files=sources, scope='Text intelligibility only; not the official model-as-judge win rate. Baseline audio is synthetic and is not a human reference.'), args.panel_size, args.shard_size))

    import gdown
    archive = cache / 'seedtts_testset.tar'
    if not archive.exists():
        temporary = str(archive) + '.incomplete'
        gdown.download(id='1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP', output=temporary, quiet=False, resume=True)
        Path(temporary).replace(archive)
    rows, meta_sha = seed_rows(archive)
    manifests.append(write_dataset(destination, 'seedtts_en', rows, dict(
        source_url='https://github.com/BytedanceSpeech/seed-tts-eval',
        publisher_code_revision='752f4297f090c46bb1a55a1f7439e5944ddefe8d', split='en/meta.lst',
        download_url='https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/view',
        archive_sha256=digest(archive), metadata_sha256=meta_sha,
        scope='Published English target texts; fixed provider voice/reference. Not a reproduction of the zero-shot speaker identity protocol.'), args.panel_size, args.shard_size))

    for corpus, resource, split, expected_md5 in [
        ('libritts', 60, 'test-clean', '7bed3bdb047c4c197f1ad3bc412db59f'),
        ('librispeech', 12, 'test-clean', '32fa31d27d2e1cad72775fee3f4849a9'),
        ('librispeech', 12, 'test-other', 'fb5a50374b501bb3bac4815ee91d3135'),
    ]:
        url = f'https://www.openslr.org/resources/{resource}/{split}.tar.gz'
        archive = download(url, cache / f'{corpus}-{split}.tar.gz')
        if digest(archive, 'md5') != expected_md5:
            raise ValueError('OpenSLR archive MD5 mismatch: ' + str(archive))
        rows = libri_rows(archive, corpus, split)
        manifests.append(write_dataset(destination, corpus + '_' + split.replace('-', '_'), rows, dict(
            source_url=f'https://www.openslr.org/{resource}/', download_url=url, split=split,
            archive_sha256=digest(archive), publisher_md5=expected_md5, license='CC BY 4.0',
            scope='TTS synthesis from official test transcripts. Original speech noise conditions do not carry over to synthesized audio. LibriTTS and LibriSpeech share source books/speakers; do not treat them as independent populations.'), args.panel_size, args.shard_size))
    index = {'protocol_id': 'public-english-v2', 'asr': {
        'checkpoint': 'Systran/faster-whisper-large-v3', 'revision': 'edaa852ec7e145841d8ffdb056a99866b5f0a478',
        'device': 'cuda', 'compute_type': 'float16', 'language': 'en', 'beam_size': 5,
        'temperature': 0, 'condition_on_previous_text': False, 'vad_filter': False},
        'datasets': manifests, 'panel_size': args.panel_size, 'seed': 42, 'repeats': 1}
    (destination / 'suite.json').write_text(json.dumps(index, indent=2) + '\n')
    print('FROZEN', sum(d['total_samples'] for d in manifests), 'English test texts', flush=True)


if __name__ == '__main__':
    main()
