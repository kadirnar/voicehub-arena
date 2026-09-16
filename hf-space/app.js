'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = x => Number.isFinite(x) ? (100*x).toFixed(2)+'%' : '—';
const num = (x,n=2) => Number.isFinite(x) ? x.toFixed(n) : '—';
const state = {data:null,sort:'wer',direction:1,page:1,pageSize:25,modelRows:new Map(),sequence:0,tab:'leaderboard',audioRows:new Map(),audioURLs:new Set(),audioRequests:new Set()};
const core = [
 ['wer','WER','Word error rate; 95% interval below',pct],
 ['cer','CER','Character error rate; 95% interval below',pct],
 ['rtf','RTF','Synthesis seconds / generated audio seconds',x=>num(x,3)],
 ['latency_p50_s','p50 · s','Median synthesis latency, seconds',x=>num(x)],
 ['latency_p95_s','p95 · s','95th percentile synthesis latency, seconds',x=>num(x)],
 ['peak_vram_mib','CUDA · GiB','Peak per-sample CUDA allocation, not process VRAM',x=>num(x/1024)],
 ['scored','Samples','All evaluated texts, including explicitly penalized no-audio failures',x=>x.toLocaleString('en-US')]
];
const extra = [
 ['mer','MER','Match error rate; lower is better',pct],['wil','WIL','Word information lost; lower is better',pct],
 ['wip','WIP','Word information preserved; higher is better',pct],['exact_match_rate','Exact match','Normalized exact transcript matches; higher is better',pct],
 ['utterance_mean_wer','Mean WER','Unweighted utterance mean, distinct from corpus WER',pct],
 ['utterance_mean_cer','Mean CER','Unweighted utterance mean, distinct from corpus CER',pct],
 ['generation_failure_rate','No audio · %','No speech was generated; these targets receive an explicit deletion penalty',pct],
 ['audio_seconds','Audio · h','Total generated audio hours',x=>num(x/3600)],
 ['silence_ratio','Silence','Mean measured silence ratio',pct],['clipping_ratio','Clipping','Mean measured clipping ratio',pct],
 ['rms_dbfs','RMS · dBFS','Mean measured RMS amplitude',x=>num(x)],['empty_asr_transcripts','Empty ASR','Empty transcripts; not proof of silent audio',x=>Number.isFinite(x)?String(x):'—']
];
function columns(){return $('columns').value==='all'?[...core,...extra]:core;}
function checkpointLink(r){
 const checkpoint=r.upstream_checkpoint||r.checkpoint,revision=r.upstream_revision||r.revision;
 const remote=/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(checkpoint)&&!checkpoint.startsWith('artifacts/');
 const suffix=/^[a-f0-9]{40}$/.test(revision)?'/tree/'+revision:'';
 return remote?`<a class="checkpoint" href="https://huggingface.co/${checkpoint}${suffix}" target="_blank" rel="noreferrer" title="${esc(checkpoint)} · ${esc(revision||'revision in manifest')}" aria-label="${esc(r.name)} checkpoint">↗</a>`:`<span class="checkpoint" title="Frozen checkpoint: ${esc(r.checkpoint)}">ⓘ</span>`;
}
function renderTable(){
 if(!state.data)return;
 const q=$('model-search').value.trim().toLowerCase();
 const rows=state.data.table.filter(r=>`${r.name} ${r.checkpoint} ${r.model}`.toLowerCase().includes(q)).sort((a,b)=>{
  const av=a[state.sort],bv=b[state.sort];
  return state.direction*(typeof av==='string'?av.localeCompare(bv):(av??Infinity)-(bv??Infinity))||a.name.localeCompare(b.name);
 });
 const cols=columns();
 $('leaderboard').tHead.innerHTML='<tr><th scope="col">#</th><th scope="col" aria-sort="'+(state.sort==='name'?(state.direction===1?'ascending':'descending'):'none')+'"><button data-sort="name">Model '+(state.sort==='name'?(state.direction===1?'↑':'↓'):'↕')+'</button></th>'+cols.map(([key,title,tip])=>`<th scope="col" aria-sort="${state.sort===key?(state.direction===1?'ascending':'descending'):'none'}"><button data-sort="${key}" title="${tip}">${title} ${state.sort===key?(state.direction===1?'↑':'↓'):'↕'}</button></th>`).join('')+'</tr>';
 const ranking=state.data.table.filter(r=>r.score_status!=='invalidated_by_implementation_bug').sort((a,b)=>a.wer-b.wer||a.name.localeCompare(b.name));
 const rank=new Map();let previous=null,currentRank=0;
 ranking.forEach((r,i)=>{if(r.wer!==previous)currentRank=i+1;rank.set(r.model,currentRank);previous=r.wer;});
 $('leaderboard').tBodies[0].innerHTML=rows.map(r=>`<tr class="${rank.get(r.model)<=3?'best':''}"><td>${rank.get(r.model)??'—'}</td><td><button class="model-name" data-model="${r.model}">${esc(r.name)}</button>${checkpointLink(r)}${r.experiment?'<span class="review-badge" title="Fixed-reference conditioning; original configuration is retained separately">Reference</span>':''}${r.generation_failures?`<span class="review-badge" title="Included in corpus WER as empty-output deletions">${r.generation_failures} no audio</span>`:''}${r.quality_review?`<span class="review-badge" title="${esc(r.quality_review_reason||'High transcript error rate; root cause unresolved')}">Review</span>`:''}</td>${cols.map(([key,,tip,format])=>`<td class="${key==='wer'||key==='cer'?'main-metric':''}" title="${esc(tip)}">${format(r[key])}${key==='wer'||key==='cer'?`<span class="ci">${(r[key+'_ci95'][0]*100).toFixed(2)}–${(r[key+'_ci95'][1]*100).toFixed(2)}</span>`:''}</td>`).join('')}</tr>`).join('')||`<tr><td colspan="${cols.length+2}" class="empty">No models match this search.</td></tr>`;
 $('model-count').textContent=`${rows.length} / ${state.data.table.length} configurations · 1,088 texts each`;
}
function showTab(tab){
 if(!['leaderboard','samples','charts'].includes(tab))return;
 state.tab=tab;
 document.querySelectorAll('[data-tab]').forEach(el=>el.setAttribute('aria-selected',String(el.dataset.tab===tab)));
 for(const name of ['leaderboard','samples','charts'])$('panel-'+name).hidden=name!==tab;
 if(tab==='samples'&&state.data)renderSamples();
}
async function loadModel(model){
 if(!state.modelRows.has(model)){
  const modelPath=state.data.table.find(r=>r.model===model)?.records_path||`data/models/${encodeURIComponent(model)}.json`;
  const promise=fetch(modelPath,{cache:'no-store'}).then(async res=>{
   if(!res.ok)throw new Error(`Could not load ${model} samples (${res.status}).`);
   const data=await res.json();if(data.rows.length!==1088)throw new Error(`Incomplete sample file for ${model}.`);return data.rows;
  }).catch(e=>{state.modelRows.delete(model);throw e;});
  state.modelRows.set(model,promise);
 }
 return state.modelRows.get(model);
}
function audioURL(row){
 const base=`https://huggingface.co/datasets/${row.audio_dataset_id||state.data.dataset_id}/resolve/${row.audio_dataset_revision||state.data.dataset_revision}/`;
 return base+row.audio_archive.split('/').map(encodeURIComponent).join('/');
}
function recording(row,model,label){
 if(!row)return '<div class="recording">Matching sample unavailable.</div>';
 const name=state.data.table.find(m=>m.model===model).name,key=model+':'+row.id;state.audioRows.set(key,row);
 if(row.status==='generation_failed_scored')return `<div class="recording"><b>${esc(name)} · No audio generated</b><p>WER ${pct(row.metrics.wer)} · CER ${pct(row.metrics.cer)} · Empty-output deletion penalty. No recording exists.</p><p>${esc(row.error)}</p></div>`;
 return `<div class="recording" data-audio-key="${esc(key)}"><div class="recording-title"><b>${label} · ${esc(name)}</b><span>WER ${pct(row.metrics.wer)} · CER ${pct(row.metrics.cer)}</span></div><p>${row.transcript.trim()?esc(row.transcript):'<em>Empty ASR transcript</em>'}</p><button class="button secondary load-audio" data-load-audio aria-label="Load ${esc(name)} audio">▶ Load audio</button><audio controls preload="none" aria-label="${esc(name)} · ${esc(row.id)}" hidden></audio><div class="audio-links"><span>${num(row.duration_s,1)} s audio · ${num(row.latency_s,2)} s synthesis</span><button class="wav-download" data-download-audio>WAV ↓</button></div><div class="audio-error" hidden>Audio could not be loaded. Please try again.</div></div>`;
}
function clearAudio(){
 for(const request of state.audioRequests)request.abort();state.audioRequests.clear();
 for(const url of state.audioURLs)URL.revokeObjectURL(url);state.audioURLs.clear();state.audioRows.clear();
}
async function loadRecording(container,download){
 const row=state.audioRows.get(container.dataset.audioKey),audio=container.querySelector('audio'),button=container.querySelector('[data-load-audio]');
 if(!row)return;
 const request=new AbortController();state.audioRequests.add(request);
 const error=container.querySelector('.audio-error');error.hidden=true;button.disabled=true;button.textContent='Loading…';
 try{
  if(!audio.src){
   const response=await fetch(audioURL(row),{headers:{Range:`bytes=${row.audio_offset}-${row.audio_offset+row.audio_bytes-1}`},signal:request.signal});
   if(response.status!==206){if(response.body)await response.body.cancel();throw new Error('The audio host did not return the requested byte range.');}
   const bytes=await response.arrayBuffer();if(bytes.byteLength!==row.audio_bytes)throw new Error('Incomplete audio response.');
   const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
   if(digest!==row.audio_sha256)throw new Error('Audio checksum mismatch.');
   if(!container.isConnected)return;
   const url=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));state.audioURLs.add(url);audio.src=url;audio.hidden=false;button.hidden=true;
  }
  if(download){const link=document.createElement('a');link.href=audio.src;link.download=row.audio_path.split('/').pop();link.click();}
  else await audio.play().catch(()=>{});
 }catch(e){if(e.name!=='AbortError'){error.textContent=e.message+' Please try again.';error.hidden=false;}}
 finally{state.audioRequests.delete(request);button.disabled=false;button.textContent='▶ Load audio';}
}
$('sample-list').addEventListener('click',e=>{const button=e.target.closest('[data-load-audio],[data-download-audio]');if(button)loadRecording(button.closest('.recording'),button.hasAttribute('data-download-audio'));});
async function renderSamples(){
 const seq=++state.sequence,model=$('model-a').value,compare=$('model-b').value;
 if(!model)return;
 $('sample-status').textContent='Loading samples…';
 clearAudio();$('sample-list').innerHTML='';
 $('previous').disabled=$('next').disabled=true;
 try{
  const [all,other]=await Promise.all([loadModel(model),compare?loadModel(compare):Promise.resolve([])]);
  if(seq!==state.sequence)return;
  const byId=new Map(other.map(r=>[r.id,r]));
  const q=$('sample-search').value.trim().toLowerCase();
  const rows=all.filter(r=>(!$('errors-only').checked||r.metrics.wer>0)&&(!q||`${r.id} ${r.reference} ${r.transcript} ${byId.get(r.id)?.transcript??''}`.toLowerCase().includes(q)));
  const pages=Math.max(1,Math.ceil(rows.length/state.pageSize));state.page=Math.max(1,Math.min(state.page,pages));
  const start=(state.page-1)*state.pageSize;
  $('sample-list').innerHTML=rows.slice(start,start+state.pageSize).map((r,i)=>`<article class="sample-card" data-sample-id="${esc(r.id)}"><div class="sample-target"><div class="meta"><span>TARGET TEXT · ${start+i+1} OF ${rows.length.toLocaleString('en-US')}</span><span>${esc(r.id)}</span></div><p>${esc(r.reference)}</p></div><div class="sample-recordings" style="--columns:${compare?2:1}">${recording(r,model,'A')}${compare?recording(byId.get(r.id),compare,'B'):''}</div></article>`).join('')||'<p class="empty">No samples match these filters.</p>';
  $('sample-count').textContent=`${rows.length.toLocaleString('en-US')} / 1,088 texts · ${rows.length?start+1:0}–${Math.min(start+state.pageSize,rows.length)} shown`;
  $('page').value=state.page;$('page').max=pages;$('page-total').textContent=`of ${pages}`;
  $('previous').disabled=state.page===1;$('next').disabled=state.page===pages;
  $('model-json').href=state.data.table.find(r=>r.model===model)?.records_path||`data/models/${model}.json`;
  $('sample-status').textContent='';
  document.querySelectorAll('audio').forEach(audio=>{
   audio.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause();}));
   audio.addEventListener('error',()=>{audio.closest('.recording').querySelector('.audio-error').hidden=false;});
  });
 }catch(e){if(seq===state.sequence){$('sample-status').textContent=e.message;$('sample-count').textContent='';}}
}
document.querySelector('.tabs').addEventListener('click',e=>{const button=e.target.closest('[data-tab]');if(button)showTab(button.dataset.tab);});
$('open-chart-exports').addEventListener('click',e=>{e.preventDefault();showTab('charts');document.querySelector('.tabs').scrollIntoView({behavior:'smooth',block:'start'});});
$('model-search').addEventListener('input',renderTable);
$('columns').addEventListener('change',renderTable);
$('leaderboard').addEventListener('click',e=>{
 const sort=e.target.closest('[data-sort]');
 if(sort){if(state.sort===sort.dataset.sort)state.direction*=-1;else{state.sort=sort.dataset.sort;state.direction=1;}renderTable();}
 const model=e.target.closest('[data-model]');
 if(model){$('model-a').value=model.dataset.model;state.page=1;showTab('samples');document.querySelector('.tabs').scrollIntoView({behavior:'smooth',block:'start'});}
});
for(const id of ['model-a','model-b','errors-only'])$(id).addEventListener('change',()=>{state.page=1;renderSamples();});
let searchTimer;
$('sample-search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.page=1;renderSamples();},160);});
$('previous').addEventListener('click',()=>{state.page--;renderSamples();});
$('next').addEventListener('click',()=>{state.page++;renderSamples();});
function goToPage(){state.page=Math.max(1,Math.trunc(Number($('page').value)||1));renderSamples();}
$('page').addEventListener('change',goToPage);
$('page').addEventListener('keydown',e=>{if(e.key==='Enter')goToPage();});
$('page-go').addEventListener('click',goToPage);
async function init(){
 try{
  const response=await fetch('data/leaderboard.json');if(!response.ok)throw new Error('Leaderboard could not be loaded. Please reload.');
  const base=await response.json();if(base.table.length!==33||base.scored_audio!==35904)throw new Error('The benchmark snapshot is incomplete.');
  let variantProgress=null;try{const p=await fetch('data/variant-progress.json',{cache:'no-store'});if(p.ok)variantProgress=await p.json();}catch{}
  const selectionResponse=await fetch('data/variant-selection.json',{cache:'no-store'});if(!selectionResponse.ok)throw Error('Active model selection is unavailable.');const selection=await selectionResponse.json();
  const data=VoiceHubCharts.mergeExperiments(base,{...variantProgress,...selection,models:variantProgress?.models||[]});state.data=data;
  $('coverage-models').textContent=data.models.toLocaleString('en-US');$('coverage-audio').textContent=data.scored_audio.toLocaleString('en-US');$('leaderboard-total').textContent=data.models;
  const failures=data.table.reduce((sum,r)=>sum+(r.generation_failures||0),0);
  const active=selection.active_model_ids.map(id=>(variantProgress?.models||[]).find(m=>m.id===id)||{id,name:id,full:{published:false,generated:0,scored:0,status:'unavailable'}});
  const completed=active.filter(m=>m.full.published&&m.full.status==='completed'&&m.full.scored===1088);
  const pending=active.filter(m=>!completed.includes(m));
  $('arena-update').textContent=variantProgress?`Last published update: ${new Date(variantProgress.updated_at*1000).toLocaleString()} · ${completed.length}/${selection.active_model_ids.length} reference experiments complete.`:'Reference progress is unavailable; showing the verified selected baseline.';
  $('variant-overview').textContent=`${completed.length}/${selection.active_model_ids.length} full reference experiments verified and included below. `+(pending.length?pending.map(m=>`${m.name}: ${m.full.generated}/1088 generated, ${m.full.scored}/1088 evaluated`).join(' · ')+'. ':'All selected experiments are complete. ')+(failures?`${failures} no-audio failures remain included in the corpus scores. `:'')+'Pilot controls and full recordings are available on the experiments page.';
  $('coverage-note').textContent=`${data.models} configurations, ${data.scored_audio.toLocaleString('en-US')} verified recordings, ${failures} explicit no-audio failures. Original and reference experiments are labeled separately.`;
  const csvFields=['model','name','protocol','checkpoint','revision','generated','generation_failures',...core.map(c=>c[0]),...extra.map(c=>c[0])];
  const cell=x=>'"'+String(x??'').replaceAll('"','""')+'"';
  const csv=[csvFields.map(cell).join(','),...data.table.map(r=>csvFields.map(k=>cell(r[k])).join(','))].join('\n');
  $('comparison-csv').href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  const choices=[...data.table].sort((a,b)=>a.name.localeCompare(b.name)).map(r=>`<option value="${r.model}">${esc(r.name)}</option>`).join('');
  $('model-a').innerHTML=choices;$('model-a').value=data.table[0].model;
  $('model-b').innerHTML='<option value="">No comparison</option>'+choices;
  $('dataset-link').href=`https://huggingface.co/datasets/${data.dataset_id}`;
  $('all-results').href=`https://huggingface.co/datasets/${data.dataset_id}/tree/main`;
  VoiceHubCharts.mount(data,model=>{$('model-a').value=model;state.page=1;showTab('samples');document.querySelector('.tabs').scrollIntoView({behavior:'smooth',block:'start'});});
  renderTable();if(state.tab==='samples')renderSamples();
  try{
   const response=await fetch('reports/current/manifest.json',{cache:'no-store'});if(!response.ok)throw Error('Export snapshot unavailable');const snapshot=await response.json();
   $('plot-snapshot').textContent=`Export snapshot: ${snapshot.configurations} configurations · ${snapshot.real_recordings.toLocaleString('en-US')} recordings · ${snapshot.no_audio_failures} no-audio failures · ${new Date(snapshot.generated_at).toLocaleString()}.`+(snapshot.configurations!==data.models?' Newer results are available in the live comparison above.':'');
  }catch(e){$('plot-snapshot').textContent='Export snapshot details are temporarily unavailable. The live table and chart remain available.';}
 }catch(e){$('load-error').textContent=e.message;$('load-error').hidden=false;$('model-count').textContent='Results unavailable';}
}
$('refresh-results').addEventListener('click',()=>window.location.reload());
init();
