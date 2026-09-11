"""Arena's explicit streaming controller over VoiceHub's native VibeVoice graph.

The 5-text / 6-speech interleave follows Microsoft's pinned streaming runtime.
This is a new Arena adapter; it is separately identified in benchmark results.
"""
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from voicehub.models.vibevoice.inference import VibeVoiceForTextToSpeech
from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.neural.cache import DynamicKVCache
from voicehub.architectures.vibevoice.codec import VibeVoiceCodecCache


class ArenaVibeVoice(VibeVoiceForTextToSpeech):
    def _validate_generation_inputs(self, inputs):
        path = inputs.get("voice_prompt_path")
        if not path or not Path(path).is_file() or Path(path).suffix != ".safetensors":
            raise ValueError("The native staged adapter requires a tensor-only voice prompt")
        limit = inputs.get("max_new_tokens")
        if limit is not None and (isinstance(limit,bool) or not isinstance(limit,int) or limit < 1):
            raise ValueError("max_new_tokens must be positive")

    def _generate(self, text, *, voice_prompt_path, output_file=None,
                  cfg_scale=1.5, max_new_tokens=1024, seed=None):
        started = time.perf_counter()
        max_new_tokens = 1024 if max_new_tokens is None else max_new_tokens
        graph = self.model
        parameter = next(graph.parameters())
        device, dtype = parameter.device, parameter.dtype
        prompt = load_file(voice_prompt_path, device=str(device))
        caches, hidden = {}, {}
        for section, layers in (("lm",4),("tts_lm",20),("neg_tts_lm",20)):
            cache = DynamicKVCache()
            for index in range(layers):
                key = prompt[f"{section}.key.{index}"].to(dtype=dtype)
                value = prompt[f"{section}.value.{index}"].to(dtype=dtype)
                cache.update(index,key,value,append=False)
            caches[section] = cache
            hidden[section] = prompt[f"{section}.hidden"].to(dtype=dtype)
        tokens = self.runtime.processor.tokenizer.encode(text.strip()+"\n").input_ids
        text_ids = torch.tensor([tokens],dtype=torch.long,device=device)
        position = 0
        generated = 0
        chunks = []
        codec_cache = VibeVoiceCodecCache()
        sample_indices = torch.tensor([0],dtype=torch.long,device=device)
        speech_id = torch.ones((1,1),dtype=torch.long,device=device)
        ttfa = None

        def mask(cache, length):
            return torch.ones((1,cache.sequence_length()+length),dtype=torch.long,device=device)

        with seeded_inference(seed,device=self.device,model_type="vibevoice") as effective_seed:
            generator = torch.Generator(device=device).manual_seed(effective_seed)
            while generated+position < max_new_tokens:
                window = text_ids[:,position:position+5]
                if window.shape[1]:
                    lower = graph.forward_lm(window,attention_mask=mask(caches["lm"],window.shape[1]),
                                             past_key_values=caches["lm"])
                    upper = graph.forward_tts_lm(window,lm_last_hidden_state=lower.last_hidden_state,
                        tts_text_masks=torch.ones((1,1),device=device,dtype=torch.long),
                        attention_mask=mask(caches["tts_lm"],window.shape[1]),past_key_values=caches["tts_lm"])
                    hidden["tts_lm"] = upper.last_hidden_state
                    position += window.shape[1]
                for _ in range(6):
                    if generated+position >= max_new_tokens:
                        break
                    latent = graph.sample_speech_latents(hidden["tts_lm"][:,-1], hidden["neg_tts_lm"][:,-1],
                        guidance_scale=cfg_scale,inference_steps=self.config.diffusion_steps,generator=generator)
                    chunk = graph.decode_speech_latents(latent,cache=codec_cache,
                        sample_indices=sample_indices,use_cache=True)
                    if ttfa is None:
                        if device.type=="cuda": torch.cuda.synchronize()
                        ttfa = time.perf_counter()-started
                    chunks.append(chunk.reshape(-1))
                    acoustic = graph.model.acoustic_connector(latent.unsqueeze(1))
                    for section in ("tts_lm","neg_tts_lm"):
                        output = graph.forward_tts_lm(speech_id,lm_last_hidden_state=acoustic,
                            tts_text_masks=torch.zeros_like(speech_id),attention_mask=mask(caches[section],1),
                            past_key_values=caches[section])
                        hidden[section] = output.last_hidden_state
                        if section=="tts_lm": positive = output
                    generated += 1
                    if torch.sigmoid(positive.logits).item() > .5:
                        if position < text_ids.shape[1]:
                            raise RuntimeError("VibeVoice ended before consuming all input text")
                        return finish_audio_output(torch.cat(chunks),self.sample_rate,output_file=output_file,
                            metadata={"runtime_adapter":"arena-native-staged-v1","seed":effective_seed,
                                      "ttfa_s":ttfa,"speech_tokens":generated,"text_tokens_consumed":position,
                                      "text_tokens_total":text_ids.shape[1]})
        raise RuntimeError("VibeVoice reached its token limit before EOS")
