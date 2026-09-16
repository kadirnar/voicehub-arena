"""Extract only the published English conditioning clips from a verified archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(archive, root):
    baseline = root / 'datasets/public/seedtts_en'
    provenance = json.loads((baseline / 'manifest.json').read_text())['provenance']
    if sha(archive) != provenance['archive_sha256']:
        raise ValueError('Publisher archive SHA256 mismatch')
    rows = [json.loads(line) for line in (baseline / 'full.jsonl').read_text().splitlines()]
    destination = root / 'datasets/native_seedtts_en'
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for row in rows:
            relative = Path(row['publisher_prompt_audio'])
            if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.wav':
                raise ValueError('Unsafe publisher audio name')
            member = tar.getmember('seedtts_testset/en/' + relative.as_posix())
            if not member.isfile():
                raise ValueError('Expected regular WAV archive member')
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            audio = tar.extractfile(member).read()
            target.write_bytes(audio)
            row.update(track='native_methods', reference_audio=str(target.relative_to(root)),
                       reference_audio_sha256=hashlib.sha256(audio).hexdigest(),
                       reference_text=row['prompt_text'])
    output = destination / 'full.jsonl'
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    manifest = dict(schema_version=1, samples=len(rows), dataset=str(output.relative_to(root)),
                    sha256=sha(output), source_dataset_sha256=sha(baseline / 'full.jsonl'),
                    source=provenance, prompt_files=len({r['reference_audio'] for r in rows}),
                    protocol='Published en/meta.lst targets and paired human prompt audio; English only; '
                             'no reference audio is appended to scored synthesis. '
                             'Whisper-large-v3 is an arena ASR choice, not an exact publisher score reproduction.')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.root), indent=2))
