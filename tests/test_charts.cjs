const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const charts=require('../hf-space/charts.js');
const raw=fs.readFileSync(path.join(__dirname,'../hf-space/data/leaderboard.json'));
const data=JSON.parse(raw);

test('all-model and custom selections retain reviewed data, while best views exclude invalidated scores',()=>{
 const top=charts.selectRows(data.table,'wer','6');
 assert.equal(top.length,6);
 assert.equal(top[0].model,'kokoro');
 assert.ok(top.every(r=>r.score_status!=='invalidated_by_implementation_bug'));
 const all=charts.selectRows(data.table,'wer','all');
 assert.equal(new Set(all.map(r=>r.model)).size,33);
 assert.equal(charts.selectRows(data.table,'wer','custom',[]).length,0);
 assert.deepEqual(charts.selectRows(data.table,'wer','custom',['dia','llasa']).map(r=>r.model),['dia','llasa']);
 const high=charts.selectRows(data.table,'exact_match_rate','6');
 assert.ok(high.every((r,i)=>i===0||high[i-1].exact_match_rate>=r.exact_match_rate));
});

test('axes contain every confidence bound and start from zero without fabricated uncertainty',()=>{
 for(const [key,m] of Object.entries(charts.metrics)){
  const rows=charts.selectRows(data.table,key,'all'),extent=charts.chartExtent(rows,key);
  assert.ok(extent.max>0&&extent.step>0);
  assert.ok(rows.every(r=>extent.max>=(m.ci?r[m.ci][1]:r[key])*m.factor));
 }
 const invalid=data.table.filter(r=>r.score_status==='invalidated_by_implementation_bug');
 if(invalid.length){
  const svg=charts.svgChart(invalid,'wer',{});
  assert.ok(svg.includes('url(#invalid-hatch)'));
  assert.ok(svg.includes('invalidated by an implementation bug'));
 }
 const svg=charts.svgChart(data.table.slice(0,2),'rtf',{kokoro:'#123456',supertonic:'#654321'});
 assert.ok(svg.includes('No confidence intervals are available'));
 assert.ok(!svg.includes('stroke-opacity=".5"'));
});

test('download plots use the same exact data and units as the live table',()=>{
 const m=JSON.parse(fs.readFileSync(path.join(__dirname,'../hf-space/reports/bars/manifest.json')));
 assert.equal(m.source_sha256,crypto.createHash('sha256').update(raw).digest('hex'));
 assert.equal(m.charts.length,8);
 for(const chart of m.charts){
  assert.equal(chart.rows.length,chart.scope==='all'?33:6);
  const factor=chart.unit==='%'?100:chart.unit==='GiB'?1/1024:1;
  for(const r of chart.rows){
   const source=data.table.find(s=>s.model===r.model);
   assert.equal(r.value,source[chart.metric]*factor);
   assert.deepEqual(r.ci95,chart.metric==='wer'||chart.metric==='cer'?source[chart.metric+'_ci95'].map(v=>v*factor):null);
  }
 }
});

test('compact comparison contains every selected model once with shared axes and bounded width',()=>{
 const rows=charts.selectRows(data.table,'wer','all');
 const svg=charts.compactChart(rows,'wer',{},2);
 assert.match(svg,/viewBox="0 0 1240 \d+"/);
 assert.equal((svg.match(/data-chart-model=/g)||[]).length,rows.length);
 for(const row of rows)assert.equal((svg.match(new RegExp('data-chart-model="'+row.model+'"','g'))||[]).length,1);
 assert.match(svg,/SAME SCALE IN EVERY COLUMN/);
 assert.match(svg,/ZERO BASELINE/);
 assert.match(svg,/95% prompt bootstrap/);
 assert.match(charts.compactChart(rows,'wer',{},1),/viewBox="0 0 620 \d+"/);
 const rtf=charts.compactChart(rows,'rtf',{},2);
 assert.match(rtf,/No confidence intervals measured/);
});

test('combined comparison excludes pilots, partial or unpublished runs and retains original measurements',()=>{
 const spec={id:'llasa-8b',name:'Llasa-8B',repo:'HKUSTAudio/Llasa-8B',revision:'a'.repeat(40),full:{published:true,status:'completed',expected:1088,scored:1088,records_path:'data/variants/full/llasa-8b.json',metrics:{wer:.02,cer:.01,scored:1088}}};
 const merged=charts.mergeExperiments(data,{models:[spec]});
 assert.equal(merged.table.length,34);assert.equal(merged.scored_audio,data.scored_audio+1088);
 assert.equal(merged.table.at(-1).records_path,spec.full.records_path);
 for(const [i,row] of data.table.entries())assert.equal(merged.table[i].wer,row.wer);
 for(const full of [{...spec.full,published:false},{...spec.full,status:'partial'},{...spec.full,scored:8}])assert.equal(charts.mergeExperiments(data,{models:[{...spec,full}]}).table.length,33);
 assert.equal(data.table.length,33);
});

test('user scope removes multilingual from all active comparisons, including stale progress',()=>{
 const selection=JSON.parse(fs.readFileSync(path.join(__dirname,'../configs/variant-selection.json')));
 const removed={id:'llasa-1b-multilingual',name:'Removed',full:{published:true,status:'completed',scored:1088,expected:1088,metrics:{wer:0,scored:1088}}};
 const merged=charts.mergeExperiments(data,{...selection,models:[removed]});
 assert.equal(merged.table.length,32);assert.equal(merged.scored_audio,32*1088);
 assert.ok(!merged.table.some(r=>r.model.startsWith('llasa')));
 assert.equal(data.table.length,33);
 const active=JSON.parse(fs.readFileSync(path.join(__dirname,'../hf-space/reports/selected/leaderboard.json')));
 assert.deepEqual(merged.table.map(r=>r.model),active.table.map(r=>r.model));
});

test('current exports and live comparison include exactly the same verified configurations and failures',()=>{
 const read=name=>fs.readFileSync(path.join(__dirname,'../hf-space/',name));
 const currentRaw=read('data/current-comparison.json'),current=JSON.parse(currentRaw);
 const progress=JSON.parse(read('data/variant-progress.json')),selection=JSON.parse(read('data/variant-selection.json'));
 const live=charts.mergeExperiments(data,{...progress,...selection});
 const snapshot=JSON.parse(read('reports/current/manifest.json'));
 const exported=JSON.parse(read('reports/current/bars/manifest.json'));
 const hash=crypto.createHash('sha256').update(currentRaw).digest('hex');
 assert.equal(snapshot.source_sha256,hash);assert.equal(exported.source_sha256,hash);
 assert.deepEqual(live.table.map(r=>r.model),current.table.map(r=>r.model));
 assert.deepEqual(snapshot.model_ids,current.table.map(r=>r.model));
 assert.equal(snapshot.real_recordings,live.scored_audio);
 assert.equal(snapshot.no_audio_failures,live.table.reduce((sum,r)=>sum+(r.generation_failures||0),0));
 for(const chart of exported.charts){
  const key=chart.metric,factor=chart.unit==='%'?100:chart.unit==='GiB'?1/1024:1;
  assert.equal(chart.rows.length,chart.scope==='all'?current.table.length:6);
  for(const row of chart.rows){const source=live.table.find(r=>r.model===row.model);assert.equal(row.value,source[key]*factor);}
 }
 const failed=live.table.filter(r=>r.generation_failures);
 for(const row of failed){
  const svg=charts.compactChart([row],'wer',{});
  assert.ok(svg.includes(`${row.generation_failures} no-audio failures included in corpus scores`));
  assert.ok(svg.includes('‡'));assert.ok(svg.includes((row.wer*100).toFixed(2)+'%'));
 }
});
