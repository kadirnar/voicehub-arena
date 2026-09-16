'use strict';
let nativeData=null;
const el=id=>document.getElementById(id);
const value=(s,key)=>typeof s?.[key]==='object'?s[key]?.value:s?.[key];
const fmt=(v,key)=>typeof v==='number'&&Number.isFinite(v)?(['wer','cer'].includes(key)?(v*100).toFixed(2)+'%':v.toFixed(3)):'—';
const label=x=>x.family+' · '+x.method.replaceAll('_',' ')+(x.streaming?' · streaming':'');
const interval=(s,key)=>s?.[key]?.ci95??s?.[key+'_ci95'];
const cell=(row,text,cls)=>{const c=document.createElement('td');c.textContent=text;if(cls)c.className=cls;row.append(c);return c;};
function render(){
 if(!nativeData)return;
 const phase=el('phase').value,key=el('metric').value,query=el('search').value.toLowerCase();
 const items=nativeData.experiments.filter(x=>(label(x)+' '+x.repo).toLowerCase().includes(query));
 el('families').textContent=new Set(nativeData.experiments.map(x=>x.repo+'@'+x.revision)).size;
 el('candidates').textContent=nativeData.experiments.length;
 el('finished').textContent=nativeData.experiments.filter(x=>x.full.published).length;
 el('audio-count').textContent=nativeData.experiments.reduce((n,x)=>n+x.full.generated+x.pilot.generated,0).toLocaleString();
 const chartItems=items.filter(x=>x[phase].published&&Number.isFinite(value(x[phase].summary,key)));
 const descending=!['wer','cer','rtf','latency_p50_s'].includes(key);
 chartItems.sort((a,b)=>(value(a[phase].summary,key)-value(b[phase].summary,key))*(descending?-1:1));
 el('coverage-note').textContent=(phase==='pilot'?'Diagnostic results on 8 predetermined texts. These are not full benchmark results. ':'All 1,088 target texts; completed evaluations only. ')+chartItems.length+' comparable methods shown. Bars start at zero; whiskers show 95% bootstrap intervals when available. Missing and inapplicable scores are excluded.';
 el('chart').replaceChildren();
 if(!chartItems.length){const box=document.createElement('div');box.className='empty';box.textContent='No verified results for this metric and coverage yet. Progress is shown below.';el('chart').append(box);}
 const bounds=chartItems.flatMap(x=>[value(x[phase].summary,key),...(interval(x[phase].summary,key)??[])]);
 const min=Math.min(0,...bounds),max=Math.max(.001,...bounds),span=max-min;
 const position=v=>(v-min)/span*100;
 for(const [i,x] of chartItems.entries()){
  const row=document.createElement('div');row.className='barrow';
  const name=document.createElement('span');name.className='barlabel';name.textContent=label(x);
  const v=value(x[phase].summary,key),ci=interval(x[phase].summary,key);
  const track=document.createElement('div');track.className='track';const bar=document.createElement('div');bar.className='bar';bar.style.left=position(Math.min(0,v))+'%';bar.style.width=(Math.abs(v)/span*100)+'%';bar.style.background=['#3546aa','#098e78','#4285db','#7758bd','#b77437'][i%5];track.append(bar);
  if(min<0){const zero=document.createElement('span');zero.className='zero';zero.style.left=position(0)+'%';track.append(zero);}
  if(ci?.length===2){const whisker=document.createElement('span');whisker.className='whisker';whisker.style.left=position(ci[0])+'%';whisker.style.width=((ci[1]-ci[0])/span*100)+'%';track.append(whisker);track.title='95% CI: '+ci.map(n=>fmt(n,key)).join(' to ');}
  const number=document.createElement('span');number.className='value';number.textContent=fmt(value(x[phase].summary,key),key);row.append(name,track,number);el('chart').append(row);
 }
 el('rows').replaceChildren();
 for(const x of items){const r=x[phase],s=r.published?r.summary:null,row=document.createElement('tr');
  const name=cell(row,x.family);const a=document.createElement('a');a.href='https://huggingface.co/'+x.repo;a.textContent=x.repo;name.append(document.createElement('br'),a);
  cell(row,x.method.replaceAll('_',' ')+(x.streaming?' · stream':''));
  const status=cell(row,r.published?'Verified':r.status==='pending'?x.implementation_status.replaceAll('_',' '):r.status.replaceAll('_',' '));
  if(x.method_note){const note=document.createElement('details');const title=document.createElement('summary');title.textContent='API support';const p=document.createElement('p');p.textContent=x.method_note;note.append(title,p);status.append(note);}
  if(r.error){const d=document.createElement('details');const title=document.createElement('summary');title.textContent='Details';const p=document.createElement('p');p.textContent=r.error;d.append(title,p);status.append(d);}
  cell(row,r.generated+' / '+r.expected,'num');
  for(const k of ['wer','cer','dnsmos_ovrl','utmos22_mos','wavlm_sim_similarity','rtf'])cell(row,k==='wavlm_sim_similarity'&&!x.uses_reference?'N/A':fmt(value(s,k),k),'num');
  const outputs=cell(row,'');if(r.published){const button=document.createElement('button');button.textContent='Listen / inspect';button.onclick=()=>loadSamples(x,phase);outputs.append(button);}else outputs.textContent='Pending';el('rows').append(row);
 }
 el('updated').textContent='Published update: '+new Date(nativeData.updated_at*1000).toLocaleString()+'. Historical VoiceHub results remain on the main arena.';
}
async function loadSamples(entry,phase){
 const response=await fetch(entry[phase].records_path);if(!response.ok)throw new Error('Sample records unavailable');
 const records=await response.json();el('samples').style.display='block';el('sample-title').textContent=label(entry)+' · '+phase;
 el('sample-select').replaceChildren();records.rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+'. '+r.text;el('sample-select').append(o);});
 const show=()=>showSample(records.rows[+el('sample-select').value]);el('sample-select').onchange=show;show();el('samples').scrollIntoView({behavior:'smooth'});
}
function showSample(row){
 const root=el('sample-detail');root.replaceChildren();const box=document.createElement('div');box.className='sample';
 for(const [name,text] of [['Target',row.reference],['Whisper transcript',row.scores.asr?.transcript??'Pending'],['Reference transcript',row.reference_text]]){const p=document.createElement('p');const b=document.createElement('strong');b.textContent=name+': ';p.append(b,document.createTextNode(text));box.append(p);}
 const metrics=document.createElement('p');metrics.className='metricline';metrics.textContent='WER '+fmt(row.scores.asr?.wer,'wer')+' · CER '+fmt(row.scores.asr?.cer,'cer')+' · DNSMOS '+fmt(row.scores.dnsmos?.ovrl,'mos')+' · UTMOS22 '+fmt(row.scores.utmos22?.mos,'mos')+' · SIM '+fmt(row.scores.wavlm_sim?.similarity,'sim');box.append(metrics);
 if(row.generation_status==='ok'){
  const button=document.createElement('button');button.textContent='Load generated audio';
  button.onclick=async()=>{button.disabled=true;try{
   const response=await fetch('https://huggingface.co/datasets/'+row.audio_dataset_id+'/resolve/'+row.audio_dataset_revision+'/'+row.audio_archive,{headers:{Range:'bytes='+row.audio_offset+'-'+(row.audio_offset+row.audio_bytes-1)}});
   if(response.status!==206)throw new Error('Server did not return the requested audio range');
   const bytes=await response.arrayBuffer();const sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
   if(sha!==row.audio_sha256)throw new Error('Audio integrity check failed');
   const audio=document.createElement('audio');audio.controls=true;audio.src=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));box.append(audio);button.remove();
  }catch(error){button.textContent=error.message;button.disabled=false;}};box.append(button);
 }else{const p=document.createElement('p');p.className='warn';p.textContent='No speech was generated. No audio file exists; the empty transcript contributes deletion errors.';box.append(p);}
 root.append(box);
}
async function refresh(){try{const r=await fetch('data/native-progress.json?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw new Error('Progress has not been published yet');nativeData=await r.json();render();}catch(error){el('updated').textContent=error.message;}}
['phase','metric','search'].forEach(id=>el(id).addEventListener(id==='search'?'input':'change',render));el('refresh').onclick=refresh;refresh();
