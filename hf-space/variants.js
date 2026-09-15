'use strict';
const el=id=>document.getElementById(id),escapeHTML=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=x=>Number.isFinite(x)?(100*x).toFixed(2)+'%':'—';
let progress,records=[],page=0,selectionSequence=0;
const downloads=new Map(),audioURLs=new Set();
function chart(){
 const rows=progress.models.filter(m=>m.full.published).map(m=>({...m.full.metrics,model:m.id,name:m.name}));
 el('variant-full-chart').innerHTML=rows.length?VoiceHubCharts.compactChart(VoiceHubCharts.selectRows(rows,el('variant-metric').value,'all'),el('variant-metric').value,{},el('variant-full-chart').clientWidth>850?2:1):'<p class="empty">Full runs are in progress. Scores will appear after all 1,088 targets are evaluated and verified.</p>';
}
function phaseStatus(p){return p.published?`Verified · ${p.scored}/${p.expected}${p.generation_failures?` · ${p.generation_failures} no audio`:''}`:`${escapeHTML(p.status.replaceAll('_',' '))} · ${p.scored}/${p.expected} scored${p.generated>p.scored?` · ${p.generated} generated`:''}${p.error?`<span class="ci">${escapeHTML(p.error)}</span>`:''}`;}
async function refresh(){
 try{
  const response=await fetch('data/variant-progress.json',{cache:'no-store'});if(!response.ok)throw Error('Progress snapshot is not available yet.');progress=await response.json();
  const selectionResponse=await fetch('data/variant-selection.json',{cache:'no-store'});if(!selectionResponse.ok)throw Error('Active model selection is unavailable.');const selection=await selectionResponse.json();progress.models=selection.active_model_ids.map(id=>progress.models.find(m=>m.id===id)).filter(Boolean);
  el('variant-error').hidden=true;
  el('variant-status').textContent=`${progress.models.filter(m=>m.full.published).length}/${progress.models.length} complete full experiments · Last published update: ${new Date(progress.updated_at*1000).toLocaleString()} · Refresh for the latest snapshot.`;
  el('variant-table').tBodies[0].innerHTML=progress.models.map(m=>`<tr><th scope="row"><a href="https://huggingface.co/${escapeHTML(m.repo)}/tree/${m.revision}" target="_blank" rel="noreferrer">${escapeHTML(m.name)} ↗</a><span class="ci">${m.revision.slice(0,10)} · fixed reference</span></th><td>${phaseStatus(m.pilot)}${m.pilot.published?`<span class="ci">Diagnostic WER ${percent(m.pilot.metrics.wer)}</span>`:''}</td><td>${phaseStatus(m.full)}</td><td>${percent(m.full.metrics?.wer)}</td><td>${percent(m.full.metrics?.cer)}</td><td>${['full','pilot'].filter(p=>m[p].published).map(p=>`<button class="model-name" data-experiment="${p}/${m.id}">${p==='full'?'Full recordings':'Pilot recordings'}</button>`).join('<br>')||'Pending verification'}</td></tr>`).join('');
  el('variant-controls').innerHTML=progress.models.filter(m=>m.unconditioned_control&&m.pilot.published).map(m=>`<article class="sample-card"><div class="sample-target"><h3>${escapeHTML(m.name)} · 8 paired texts</h3><p><b>Without reference: ${percent(m.unconditioned_control.metrics.wer)} WER</b> → <b>With reference: ${percent(m.pilot.metrics.wer)} WER</b></p><p>Corpus word edits: ${m.unconditioned_control.metrics.word_substitutions+m.unconditioned_control.metrics.word_deletions+m.unconditioned_control.metrics.word_insertions}/${m.unconditioned_control.metrics.reference_words} → ${m.pilot.metrics.word_substitutions+m.pilot.metrics.word_deletions+m.pilot.metrics.word_insertions}/${m.pilot.metrics.reference_words}. Diagnostic only; not the full 1,088-text result.</p><button class="button secondary" data-experiment="pilot/${m.id}-unconditioned">Listen without reference</button> <button class="button secondary" data-experiment="pilot/${m.id}">Listen with reference</button></div></article>`).join('')||'<p>Paired controls are being verified.</p>';
  const old=el('variant-samples').value;let choices=[];
  for(const m of progress.models){for(const p of ['full','pilot'])if(m[p].published)choices.push([`${p}/${m.id}`,`${m.name} · ${p==='full'?'FULL 1,088':'PILOT 8'} · reference`,m[p].records_path]);if(m.unconditioned_control)choices.push([`pilot/${m.id}-unconditioned`,`${m.name} · PILOT 8 · no reference`,m.unconditioned_control.records_path]);}
  downloads.clear();for(const [key,,path] of choices)downloads.set(key,path);
  el('variant-samples').innerHTML=choices.map(([key,label])=>`<option value="${key}">${escapeHTML(label)}</option>`).join('');
  if(downloads.has(old))el('variant-samples').value=old;
  chart();if(el('variant-samples').value!==old||!records.length)await loadSamples();
 }catch(e){el('variant-error').textContent=e.message;el('variant-error').hidden=false;}
}
async function loadSamples(){
 const seq=++selectionSequence,key=el('variant-samples').value;records=[];page=0;if(!key)return;
 el('variant-sample-status').textContent='Loading verified records…';
 try{const response=await fetch(downloads.get(key));if(!response.ok)throw Error('Could not load records');const data=await response.json();if(seq!==selectionSequence)return;records=data.rows;el('variant-sample-status').textContent=`${data.phase.toUpperCase()} · ${records.length} texts · ${data.conditioning.replaceAll('_',' ')} · WER ${percent(data.summary.wer)}`;renderSamples();}
 catch(e){el('variant-sample-status').textContent=e.message;}
}
function renderSamples(){
 for(const url of audioURLs)URL.revokeObjectURL(url);audioURLs.clear();
 el('variant-sample-list').innerHTML=records.slice(page*8,page*8+8).map((r,i)=>`<article class="sample-card"><div class="sample-target"><div class="meta">${page*8+i+1} / ${records.length} · ${escapeHTML(r.id)}</div><p>${escapeHTML(r.reference)}</p></div><div class="recording"><b>WER ${percent(r.metrics.wer)} · CER ${percent(r.metrics.cer)}</b><p>${r.transcript?escapeHTML(r.transcript):r.status==='generation_failed_scored'?'<em>No audio generated. Empty-output deletion penalty.</em>':'<em>Empty Whisper transcript</em>'}</p>${r.status==='ok'?`<button class="button secondary" data-listen="${page*8+i}">Load audio</button>`:`<p>${escapeHTML(r.error)}</p>`}<audio controls preload="none" hidden></audio><p class="audio-error" hidden></p></div></article>`).join('');
 el('variant-page').textContent=`Page ${page+1} / ${Math.max(1,Math.ceil(records.length/8))}`;el('variant-prev').disabled=page===0;el('variant-next').disabled=(page+1)*8>=records.length;
}
async function listen(button){
 const r=records[Number(button.dataset.listen)],container=button.closest('.recording'),audio=container.querySelector('audio'),error=container.querySelector('.audio-error');button.disabled=true;
 try{const response=await fetch(`https://huggingface.co/datasets/${r.audio_dataset_id}/resolve/${r.audio_dataset_revision}/${r.audio_archive}`,{headers:{Range:`bytes=${r.audio_offset}-${r.audio_offset+r.audio_bytes-1}`}});if(response.status!==206){await response.body?.cancel();throw Error('Audio range request failed');}const bytes=await response.arrayBuffer();const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');if(hash!==r.audio_sha256)throw Error('Audio hash mismatch');if(!container.isConnected)return;const url=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));audioURLs.add(url);audio.src=url;audio.hidden=false;button.hidden=true;await audio.play().catch(()=>{});}
 catch(e){error.textContent=e.message;error.hidden=false;}finally{button.disabled=false;}
}
document.addEventListener('click',async e=>{const x=e.target.closest('[data-experiment]');if(x){el('variant-samples').value=x.dataset.experiment;await loadSamples();el('variant-recordings').scrollIntoView({behavior:'smooth'});}const b=e.target.closest('[data-listen]');if(b)listen(b);});
function selectChartModel(e){const x=e.target.closest('[data-chart-model]');if(x&&(e.type==='click'||e.key==='Enter'||e.key===' ')){e.preventDefault();el('variant-samples').value='full/'+x.dataset.chartModel;loadSamples();el('variant-recordings').scrollIntoView({behavior:'smooth'});}}
el('variant-full-chart').addEventListener('click',selectChartModel);el('variant-full-chart').addEventListener('keydown',selectChartModel);
el('refresh-variants').addEventListener('click',refresh);el('variant-metric').addEventListener('change',chart);el('variant-samples').addEventListener('change',loadSamples);el('variant-prev').addEventListener('click',()=>{page--;renderSamples();});el('variant-next').addEventListener('click',()=>{page++;renderSamples();});refresh();
