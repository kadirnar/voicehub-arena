/* The chart reads the exact same snapshot as the table; it never estimates scores. */
(function(root){
 'use strict';
 const metrics={
  wer:{label:'Word error rate',short:'WER',unit:'%',factor:100,digits:2,ci:'wer_ci95'},
  cer:{label:'Character error rate',short:'CER',unit:'%',factor:100,digits:2,ci:'cer_ci95'},
  rtf:{label:'Synthesis speed',short:'RTF',unit:'×',factor:1,digits:3},
  latency_p50_s:{label:'Median synthesis latency',short:'Latency p50',unit:'s',factor:1,digits:2},
  latency_p95_s:{label:'Tail synthesis latency',short:'Latency p95',unit:'s',factor:1,digits:2},
  peak_vram_mib:{label:'Peak CUDA allocation',short:'GPU memory',unit:'GiB',factor:1/1024,digits:2},
  mer:{label:'Match error rate',short:'MER',unit:'%',factor:100,digits:2},
  wil:{label:'Word information lost',short:'WIL',unit:'%',factor:100,digits:2},
  exact_match_rate:{label:'Exact transcript match',short:'Exact match',unit:'%',factor:100,digits:2,higher:true}
 };
 const palette=['#32369a','#0fa383','#4385ee','#bb754a','#9771bd','#317c39','#c65b78','#347f9e','#827344','#5868aa','#00857a','#a96797'];
 const invalid=r=>r.score_status==='invalidated_by_implementation_bug';
 const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function selectRows(rows,key,scope,selected=[]){
  const metric=metrics[key];
  let chosen=rows.filter(r=>Number.isFinite(r[key]));
  if(scope==='custom')chosen=chosen.filter(r=>selected.includes(r.model));
  else if(scope!=='all')chosen=chosen.filter(r=>!invalid(r));
  chosen.sort((a,b)=>(invalid(a)-invalid(b))+(invalid(a)===invalid(b)?((metric.higher?-1:1)*(a[key]-b[key])||a.name.localeCompare(b.name)):0));
  return scope==='6'||scope==='12'?chosen.slice(0,Number(scope)):chosen;
 }
 function chartExtent(rows,key){
  const m=metrics[key];
  const peak=Math.max(0,...rows.map(r=>Math.max(r[key],m.ci&&Array.isArray(r[m.ci])?r[m.ci][1]:r[key])*m.factor));
  const target=(peak||1)*1.2,rough=target/5,power=10**Math.floor(Math.log10(rough));
  const step=[1,2,2.5,5,10].map(n=>n*power).find(n=>n>=rough);
  return {max:Math.ceil(target/step)*step,step};
 }
 function labelLines(text){
  const words=text.replace(/ · /g,' ').split(/\s+/),lines=[''];
  for(const word of words){if(lines[lines.length-1]&&(lines[lines.length-1]+' '+word).length>19)lines.push(word);else lines[lines.length-1]+=(lines[lines.length-1]?' ':'')+word;}
  return lines;
 }
 function initials(row){return ({kokoro:'K',supertonic:'ST',omnivoice:'OV',speecht5:'S5',f5tts:'F5',styletts2:'S2',cosyvoice:'C3',qwen3tts:'Q3',mosstts:'M',zonos2:'Z2'})[row.model]||row.model.slice(0,2).toUpperCase();}
 function svgChart(rows,key,colors){
  const m=metrics[key],width=Math.max(960,rows.length*155+90),height=530,left=66,right=24,top=70,bottom=374;
  const extent=chartExtent(rows,key),plotWidth=width-left-right,slot=plotWidth/Math.max(rows.length,1),barWidth=Math.min(126,slot*.78);
  const y=v=>bottom-v/extent.max*(bottom-top),fmt=v=>v.toFixed(m.digits)+(m.unit==='%'?'%':' '+m.unit);
  let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-labelledby="bar-svg-title bar-svg-desc"><title id="bar-svg-title">${m.label}: ${rows.length} TTS models</title><desc id="bar-svg-desc">Linear axis starts at zero. ${m.higher?'Higher':'Lower'} is better. ${m.ci?'Whiskers are 95% prompt bootstrap confidence intervals.':'No confidence intervals are available for this metric.'} Each model uses all 1,088 texts. Individual values are listed below the chart.</desc><defs><pattern id="invalid-hatch" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#d9d5cb"/><path d="M-2 2L2-2M0 8L8 0M6 10L10 6" stroke="#99948b" stroke-width="1.5"/></pattern></defs><rect width="100%" height="100%" fill="#fdfcf8"/><g font-family="Inter,Arial,sans-serif"><text x="${left}" y="34" font-size="12" fill="#707789">${esc(m.short)} (${esc(m.unit)})</text><text x="${width-right}" y="34" text-anchor="end" font-size="19" font-weight="700" fill="#343891">VoiceHub <tspan fill="#717482" font-weight="400">Arena</tspan></text>`;
  for(let v=0;v<=extent.max+extent.step/100;v+=extent.step){
   const digits=extent.step<.1?2:extent.step<1?1:0;
   svg+=`<line x1="${left}" x2="${width-right}" y1="${y(v)}" y2="${y(v)}" stroke="#e6e2d8"/><text x="${left-12}" y="${y(v)+4}" text-anchor="end" fill="#858997" font-size="12">${v.toFixed(digits)}</text>`;
  }
  rows.forEach((r,i)=>{
   const x=left+slot*(i+.5),v=r[key]*m.factor,ci=m.ci&&Array.isArray(r[m.ci])?r[m.ci].map(n=>n*m.factor):null;
   const marker=invalid(r)?' †':r.quality_review?' *':'',color=invalid(r)?'#918b82':colors[r.model],high=ci?ci[1]:v;
   const explanation=invalid(r)?'Archived result invalidated by an implementation bug':r.quality_review?'Quality under review':'';
   svg+=`<g class="bar-model" data-chart-model="${esc(r.model)}" tabindex="0" role="link" aria-label="Listen to ${esc(r.name)}: ${fmt(v)}${explanation?'; '+explanation:''}"><title>${esc(r.name)}: ${fmt(v)}${ci?`; 95% CI ${fmt(ci[0])} to ${fmt(ci[1])}`:''}${explanation?'; '+explanation:''}</title><rect x="${x-barWidth/2}" y="${y(v)}" width="${barWidth}" height="${Math.max(0,bottom-y(v))}" rx="9" fill="${invalid(r)?'url(#invalid-hatch)':color}"/>`;
   if(ci)svg+=`<path d="M${x} ${y(ci[0])}V${y(ci[1])}M${x-6} ${y(ci[0])}H${x+6}M${x-6} ${y(ci[1])}H${x+6}" fill="none" stroke="#353942" stroke-opacity=".5" stroke-width="1.5"/>`;
   svg+=`<text class="bar-value" x="${x}" y="${y(high)-12}" text-anchor="middle" font-size="19" font-weight="700" fill="#252935">${fmt(v)}${marker}</text><circle cx="${x}" cy="${bottom+4}" r="23" fill="${color}" stroke="#fdfcf8" stroke-width="3"/><text x="${x}" y="${bottom+10}" text-anchor="middle" font-size="15" font-weight="600" fill="white">${initials(r)}</text>`;
   labelLines(r.name).forEach((line,j)=>{svg+=`<text x="${x}" y="${bottom+57+j*18}" text-anchor="middle" font-size="12.5" fill="#575c68">${esc(line)}${j===0?marker:''}</text>`;});
   svg+='</g>';
  });
  svg+=`<text x="${left}" y="505" font-size="11" fill="#858997">SEED-TTS-EVAL · ENGLISH · 1,088 TEXTS PER MODEL · WHISPER-LARGE-V3</text><text x="${width-right}" y="505" text-anchor="end" font-size="11" fill="#858997">${m.higher?'HIGHER':'LOWER'} IS BETTER · ZERO BASELINE</text></g></svg>`;
  return svg;
 }
 function compactChart(rows,key,colors,columns=2){
  const m=metrics[key],extent=chartExtent(rows,key),cols=rows.length>12?columns:1;
  const panel=620,width=cols*panel,count=Math.ceil(rows.length/cols),top=78,rowHeight=27,height=top+count*rowHeight+65;
  const left=225,plot=280,valueX=605,fmt=v=>v.toFixed(m.digits)+(m.unit==='%'?'%':' '+m.unit);
  let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-labelledby="compact-title compact-desc"><title id="compact-title">${esc(m.label)}: all ${rows.length} selected models</title><desc id="compact-desc">${cols} columns share one linear scale starting at zero. Read ranks down each column. ${m.higher?'Higher':'Lower'} is better. ${m.ci?'Whiskers show 95% prompt bootstrap confidence intervals.':'No confidence intervals measured.'} Every number comes from the full table.</desc><rect width="100%" height="100%" fill="#fdfcf8"/><g font-family="Inter,Arial,sans-serif">`;
  for(let col=0;col<cols;col++){
   const ox=col*panel;
   svg+=`<text x="${ox+14}" y="26" font-size="11" font-weight="600" fill="#737887">MODEL · ${col*count+1}–${Math.min(rows.length,(col+1)*count)}</text><text x="${ox+valueX}" y="26" text-anchor="end" font-size="11" font-weight="600" fill="#737887">${esc(m.short)} (${esc(m.unit)})</text>`;
   for(let v=0;v<=extent.max+extent.step/100;v+=extent.step){
    const x=ox+left+v/extent.max*plot,digits=extent.step<.1?2:extent.step<1?1:0;
    svg+=`<line x1="${x}" x2="${x}" y1="64" y2="${top+count*rowHeight-8}" stroke="#e9e5dd"/><text x="${x}" y="53" text-anchor="middle" font-size="10" fill="#858997">${v.toFixed(digits)}</text>`;
   }
  }
  rows.forEach((r,i)=>{
   const col=Math.floor(i/count),ox=col*panel,y=top+(i%count)*rowHeight,v=r[key]*m.factor,ci=m.ci&&Array.isArray(r[m.ci])?r[m.ci].map(n=>n*m.factor):null;
   const color=invalid(r)?'#918b82':colors[r.model]||palette[i%palette.length],marker=invalid(r)?' †':r.quality_review?' *':'';
   const label=r.name.length>28?r.name.slice(0,26)+'…':r.name,detail=`${r.name}: ${fmt(v)}${ci?`; 95% CI ${fmt(ci[0])}–${fmt(ci[1])}`:''}${marker?' · quality review':''}`;
   svg+=`<g class="bar-model compact-model" data-chart-model="${esc(r.model)}" tabindex="0" role="link" aria-label="${esc(detail)}; listen to samples"><title>${esc(detail)}</title><rect x="${ox+6}" y="${y-18}" width="608" height="25" rx="4" fill="${i%2?'#f4f2ed':'transparent'}"/><text x="${ox+14}" y="${y}" font-size="10.5" fill="#9297a1">${i+1}</text><text x="${ox+42}" y="${y}" font-size="11.5" font-weight="500" fill="#353a46">${esc(label)}${marker}</text><rect x="${ox+left}" y="${y-11}" width="${v/extent.max*plot}" height="10" rx="3" fill="${color}"/>`;
   if(ci){const x1=ox+left+ci[0]/extent.max*plot,x2=ox+left+ci[1]/extent.max*plot;svg+=`<path d="M${x1} ${y-6}H${x2}M${x1} ${y-10}V${y-2}M${x2} ${y-10}V${y-2}" stroke="#333945" stroke-width="1" fill="none"/>`;}
   svg+=`<text x="${ox+valueX}" y="${y}" text-anchor="end" font-size="12" font-weight="650" fill="#252935">${fmt(v)}</text></g>`;
  });
  return svg+`<text x="14" y="${height-20}" font-size="10" fill="#858997">SEED-TTS-EVAL · ENGLISH · WHISPER-LARGE-V3 · SAME SCALE IN EVERY COLUMN</text><text x="${width-15}" y="${height-20}" text-anchor="end" font-size="10" fill="#858997">${m.higher?'HIGHER':'LOWER'} IS BETTER · ZERO BASELINE</text></g></svg>`;
 }
 function mount(data,onModel){
  const $=id=>document.getElementById(id),all=data.table,colors={};
  [...all].sort((a,b)=>a.model.localeCompare(b.model)).forEach((r,i)=>colors[r.model]=palette[i%palette.length]);
  Object.entries({kokoro:palette[0],supertonic:palette[1],omnivoice:palette[2],speecht5:palette[3],f5tts:palette[4],styletts2:palette[5]}).forEach(([k,v])=>colors[k]=v);
  $('chart-metric').innerHTML=Object.entries(metrics).map(([key,m])=>`<option value="${key}">${m.short} · ${m.unit}</option>`).join('');
  const initial=selectRows(all,'wer','all').map(r=>r.model);
  $('chart-model-options').innerHTML=[...all].sort((a,b)=>a.name.localeCompare(b.name)).map(r=>`<label><input type="checkbox" value="${esc(r.model)}" ${initial.includes(r.model)?'checked':''}><span>${esc(r.name)}${invalid(r)?' †':r.quality_review?' *':''}</span></label>`).join('');
  let downloadURL=null;
  function render(){
   const key=$('chart-metric').value,scope=$('chart-scope').value,m=metrics[key],chosen=[...$('chart-model-options').querySelectorAll('input:checked')].map(x=>x.value),rows=selectRows(all,key,scope,chosen).filter(r=>!$('chart-search').value||[r.name,r.model,r.checkpoint].join(' ').toLowerCase().includes($('chart-search').value.toLowerCase()));
   $('bar-chart-title').textContent=m.label;
   $('chart-summary').textContent=`${rows.length} of ${all.length} models · ${m.higher?'Higher':'Lower'} is better${m.ci?' · 95% confidence intervals':''}`;
   $('bar-chart').classList.toggle('compact', $('chart-layout').value==='compact');
   $('bar-chart').innerHTML=rows.length?($('chart-layout').value==='compact'?compactChart(rows,key,colors,$('bar-chart').clientWidth>850?2:1):svgChart(rows,key,colors)):'<p class="empty">Select at least one model below to draw a comparison.</p>';
   $('chart-notes').textContent=(m.ci?'Whiskers: 95% prompt bootstrap intervals. ':'No confidence intervals were measured for this metric. ')+($('chart-layout').value==='compact'?'Columns use the same scale. ':'Scroll horizontally to see every bar. ')+(rows.some(invalid)?'† Archived, invalidated score; excluded from ranking. ':'')+(rows.some(r=>r.quality_review&&!invalid(r))?'* Quality under review. ':'')+`Click a model to listen. The full ${all.length}-model table is below.`;
   $('chart-values-body').innerHTML=rows.map(r=>`<tr><th scope="row">${esc(r.name)}${invalid(r)?' †':r.quality_review?' *':''}</th><td>${(r[key]*m.factor).toFixed(m.digits)} ${m.unit}</td><td>${m.ci&&r[m.ci]?r[m.ci].map(v=>(v*m.factor).toFixed(m.digits)).join('–')+' '+m.unit:'Not measured'}</td></tr>`).join('');
   if(downloadURL)URL.revokeObjectURL(downloadURL);
   const drawn=$('bar-chart').querySelector('svg');if(drawn){downloadURL=URL.createObjectURL(new Blob([drawn.outerHTML],{type:'image/svg+xml'}));$('chart-download').href=downloadURL;$('chart-download').download=`voicehub-${key}-${rows.length}-models.svg`;$('chart-download').hidden=false;}else $('chart-download').hidden=true;
   $('chart-values-metric').textContent=m.short;
   $('chart-model-count').textContent=`Choose models (${chosen.length} selected)`;
  }
  $('chart-layout').addEventListener('change',render);$('chart-search').addEventListener('input',render);
  $('chart-metric').addEventListener('change',render);$('chart-scope').addEventListener('change',()=>{if($('chart-scope').value==='custom')$('chart-model-picker').open=true;render();});
  $('chart-model-options').addEventListener('change',()=>{$('chart-scope').value='custom';render();});
  for(const [id,checked] of [['chart-select-all',true],['chart-clear',false]])$(id).addEventListener('click',()=>{$('chart-model-options').querySelectorAll('input').forEach(input=>input.checked=checked);$('chart-scope').value='custom';render();});
  function activate(e){const model=e.target.closest('[data-chart-model]');if(model&&(e.type==='click'||e.key==='Enter'||e.key===' ')){e.preventDefault();onModel(model.dataset.chartModel);}}
  $('bar-chart').addEventListener('click',activate);$('bar-chart').addEventListener('keydown',activate);
  let lastWidth=0;new ResizeObserver(entries=>{const width=entries[0].contentRect.width;if(Math.abs(width-lastWidth)>50){lastWidth=width;render();}}).observe($('bar-chart'));
  render();
 }
 const api={metrics,selectRows,chartExtent,labelLines,svgChart,compactChart,mount};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.VoiceHubCharts=api;
})(typeof window==='undefined'?globalThis:window);

/* Merge only completely scored, published experiments. Preserve the base snapshot. */
(function(root){
 function mergeExperiments(base,progress){
  const table=base.table.map(r=>({...r,protocol:'Original provider configuration'}));
  const ids=new Set(table.map(r=>r.model));
  for(const spec of progress?.models||[]){
   const full=spec.full;
   if(!full?.published||full.status!=='completed'||full.scored!==1088||full.expected!==1088)continue;
   if(ids.has(spec.id))throw Error('Duplicate variant experiment ID');
   ids.add(spec.id);
   table.push({...full.metrics,model:spec.id,name:spec.name+' · ref',checkpoint:spec.repo,revision:spec.revision,protocol:'Fixed reference · independent implementation',records_path:full.records_path,experiment:'reference-conditioned'});
  }
  return {...base,table,models:table.length,scored_audio:table.reduce((sum,r)=>sum+(r.generated??r.scored),0)};
 }
 if(typeof module!=='undefined'&&module.exports)module.exports.mergeExperiments=mergeExperiments;else root.VoiceHubCharts.mergeExperiments=mergeExperiments;
})(typeof window==='undefined'?globalThis:window);
