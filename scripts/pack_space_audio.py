"""Pack verified WAVs as indexed WebDataset tar shards for range-based playback."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def pack(dataset,space):
    archive_dir=dataset/'audio_shards';archive_dir.mkdir(exist_ok=True)
    files=json.loads((dataset/'audio-manifest.json').read_text())
    expected={f['path']:f for f in files}
    shards=[];audio_index=[]
    for path in sorted((space/'data/models').glob('*.json')):
        data=json.loads(path.read_text());model=data['model'];archive=archive_dir/f'{model}.tar'
        metadata={r['id']:r for r in map(json.loads,(dataset/'audio'/model/'metadata.jsonl').read_text().splitlines())}
        with tarfile.open(archive,'w',format=tarfile.PAX_FORMAT) as tar:
            for row in data['rows']:
                audio=dataset/row['audio_path'];known=expected[row['audio_path']]
                assert audio.stat().st_size==known['size'] and row['audio_sha256']==known['sha256']
                info=tarfile.TarInfo(audio.name);info.size=audio.stat().st_size;info.mode=0o644
                offset=tar.offset+len(info.tobuf(format=tarfile.PAX_FORMAT))
                with audio.open('rb') as stream:tar.addfile(info,stream)
                row.update(audio_archive=f'audio_shards/{model}.tar',audio_offset=offset,audio_bytes=info.size)
                audio_index.append({'path':row['audio_path'],'archive':row['audio_archive'],'entry':audio.name,
                                    'offset':offset,'size':info.size,'sha256':row['audio_sha256']})
                raw=json.dumps(metadata[row['id']],ensure_ascii=False).encode()
                meta=tarfile.TarInfo(audio.stem+'.json');meta.size=len(raw);meta.mode=0o644
                tar.addfile(meta,io.BytesIO(raw))
        # Validate all entries and offsets against original synthesis hashes.
        h=hashlib.sha256()
        with archive.open('rb') as stream:
            while chunk:=stream.read(8*1024*1024):h.update(chunk)
            for row in data['rows']:
                stream.seek(row['audio_offset']);wave=stream.read(row['audio_bytes'])
                assert hashlib.sha256(wave).hexdigest()==row['audio_sha256']
        with tarfile.open(archive) as tar:
            entries=tar.getmembers();assert len(entries)==2176
            assert sum(x.name.endswith('.wav') for x in entries)==1088
        raw=json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n';path.write_text(raw)
        record=dataset/'records'/path.name;original=json.loads(record.read_text());original['rows']=data['rows']
        record.write_text(json.dumps(original,ensure_ascii=False,separators=(',',':'))+'\n')
        shards.append({'path':f'audio_shards/{archive.name}','size':archive.stat().st_size,'sha256':h.hexdigest(),'samples':1088})
        print(json.dumps(shards[-1]),flush=True)
    assert len(shards)==33 and len(audio_index)==35904
    (dataset/'audio-manifest.json').write_text(json.dumps(audio_index,separators=(',',':'))+'\n')
    (dataset/'audio-shards.json').write_text(json.dumps(shards,indent=2)+'\n')
    for root in (space/'data',dataset):
        p=root/'leaderboard.json';data=json.loads(p.read_text());data['audio_format']='indexed-wav-tar';p.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--space-dir',type=Path,required=True)
    a=p.parse_args();pack(a.dataset_dir,a.space_dir)
