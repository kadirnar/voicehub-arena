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
