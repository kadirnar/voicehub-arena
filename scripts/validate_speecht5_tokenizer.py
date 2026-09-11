"""Compare the actual pinned character tokenizer against SentencePiece C++."""
import json
from pathlib import Path
from voicehub_arena.auth import configure_hub_auth
from voicehub_arena.transport import enable_verified_cache_reuse
configure_hub_auth()
enable_verified_cache_reuse()
import sentencepiece
from voicehub.hub_transport import download_hugging_face_file
from voicehub.models.speecht5.processing import SpeechT5Tokenizer

path = download_hugging_face_file('microsoft/speecht5_tts','spm_char.model',
    revision='30fcde30f19b87502b8435427b5f5068e401d5f6')
native = SpeechT5Tokenizer(path)
reference = sentencepiece.SentencePieceProcessor(model_file=str(path))
texts = [json.loads(line)['text'] for line in Path('datasets/english.jsonl').read_text().splitlines()]
texts += ['', '  Hello\t world!  ', 'Café — naïve café.', 'Unknown 12345 😀🙂 characters.', 'e\u0301 æ œ ê']
for text in texts:
    actual = native.encode(text)
    expected = reference.encode(text,out_type=int)+[reference.eos_id()]
    if actual != expected:
        raise AssertionError((text,actual,expected))
print('Published SpeechT5 tokenizer reference parity:',len(texts),'cases passed')
