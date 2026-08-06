#!/usr/bin/env python3
"""Build a standalone fixed-window model-comparison dashboard from JSON reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from plotly.offline import get_plotlyjs


def parse_report(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Reports must be LABEL=PATH") from error
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", default=[], type=parse_report,
                        help="Completed report as LABEL=month_state_summary.json; repeatable.")
    parser.add_argument("--expected", action="append", default=[],
                        help="Expected comparison label without a report yet; repeatable.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {label: json.loads(path.read_text()) for label, path in args.report}
    labels = list(dict.fromkeys([*reports, *args.expected]))
    if not labels:
        parser.error("Supply at least one --report or --expected label")
    payload = json.dumps(reports).replace("</", "<\\/")
    label_json = json.dumps(labels).replace("</", "<\\/")
    plotly_js = get_plotlyjs().replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECH2O fixed-window recurrent model comparison</title><script>__PLOTLY_JS__</script><style>
body{font:15px system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 18px;color:#17212b;background:#fafbfd} h1{margin-bottom:4px}.muted{color:#52616d}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}label{font-weight:600}select{margin-left:6px;padding:5px}table{border-collapse:collapse;background:white;width:100%;margin:16px 0 32px}th,td{border:1px solid #d7dee5;padding:7px;text-align:right}th{background:#edf2f7}td:first-child,th:first-child{text-align:left}.pending{color:#8a5a00;background:#fff7df}.card{background:white;border:1px solid #d7dee5;border-radius:7px;padding:12px;margin:8px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:18px}.plot{background:white;border:1px solid #d7dee5;border-radius:7px;padding:12px}.chart{height:390px}</style></head>
<body><h1>ECH2O fixed-window recurrent model comparison</h1><p class="muted">Pixel-weighted spatial-validation metrics. Select a month and state to compare identical targets across completed models. Pending models are intentionally blank rather than inferred.</p><p><a href="pixel_strata_explorer.html">Open pixel-level terrain and climate strata explorer →</a></p>
<div class="controls"><label>Month <select id="month"><option value="all">All June–September</option><option value="6">June</option><option value="7">July</option><option value="8">August</option><option value="9">September</option></select></label><label>State <select id="state"><option value="all">All states</option></select></label><label>Metric <select id="metric"><option value="mae">MAE ↓</option><option value="rmse">RMSE ↓</option><option value="bias">Bias</option><option value="correlation">Correlation ↑</option><option value="r2">R² ↑</option><option value="sd_ratio">SD ratio (target 1)</option></select></label><label>Target <select id="target"></select></label><label>Site context <select id="context"><option value="wbdef_median">Climatic water deficit</option><option value="tpi_median">Topographic position (TPI)</option><option value="twi_median">Topographic wetness (TWI)</option><option value="cell_count">BBox cell count</option></select></label></div>
<div class="plots"><section class="plot"><h2>Selected target across models</h2><p class="muted">Selected target, month, state, and metric. Hover for exact values; use the Plotly toolbar to zoom or save.</p><div id="barChart" class="chart"></div></section><section class="plot"><h2>Seasonal trajectory</h2><p class="muted">Selected target across June–September. “All states” shows pooled validation metrics; a named state shows that state only.</p><div id="lineChart" class="chart"></div></section><section class="plot"><h2>Site-context diagnostic</h2><p class="muted">Each point is one validation site, pooled across June–September. Hover for site ID, state, fixed split quartile, and context value.</p><div id="contextChart" class="chart"></div></section></div>
<h2>Comparison table</h2><div id="comparison"></div><h2>Available reports</h2><div class="grid" id="cards"></div>
<script>const reports=__PAYLOAD__;const labels=__LABELS__;const targets=['soilmoisture','tskin_am','tskin_pm','plc_am','plc_pm'];const colors=['#1769aa','#d95f02','#1b9e77','#7570b3','#e7298a','#66a61e'];
const states=[...new Set(Object.values(reports).flatMap(r=>Object.values(r.by_month||{}).flatMap(m=>Object.keys(m.by_state||{}))))].sort();const stateSelect=document.querySelector('#state'),targetSelect=document.querySelector('#target'),contextSelect=document.querySelector('#context');states.forEach(s=>stateSelect.add(new Option(s,s)));targets.forEach(t=>targetSelect.add(new Option(t,t)));
function valueFor(report,month,state,target){if(!report)return null;if(month==='all'&&state==='all')return report.overall_by_target?.[target]||null;if(month==='all')return report.by_state?.[state]?.[target]||null;if(state==='all')return report.by_month?.[month]?.overall_by_target?.[target]||null;return report.by_month?.[month]?.by_state?.[state]?.[target]||null;}
function metricRange(values,metric){const finite=values.filter(v=>Number.isFinite(v));if(!finite.length)return[0,1];const lo=Math.min(...finite),hi=Math.max(...finite);if(['mae','rmse'].includes(metric))return[0,Math.max(hi*1.15,0.001)];if(metric==='bias'){const end=Math.max(Math.abs(lo),Math.abs(hi),0.001)*1.15;return[-end,end]}const spread=Math.max(hi-lo,Math.abs(hi)*0.06,0.001);return[lo-spread*.2,hi+spread*.2]}
const chartConfig={responsive:true,displaylogo:false};
function drawBars(month,state,metric,target){const active=labels.filter(x=>reports[x]),values=active.map(label=>valueFor(reports[label],month,state,target)?.[metric]),range=metricRange(values,metric);const trace={type:'bar',x:active,y:values,marker:{color:active.map((_,i)=>colors[i%colors.length])},hovertemplate:'%{x}<br>'+target+' '+metric+': %{y:.6g}<extra></extra>'};const layout={title:target+' — '+metric,yaxis:{title:metric,range:range},xaxis:{automargin:true},margin:{l:65,r:15,t:45,b:70},paper_bgcolor:'#fff',plot_bgcolor:'#fff'};Plotly.react('barChart',[trace],layout,chartConfig)}
function drawLines(state,metric,target){const months=['6','7','8','9'],names=['June','July','August','September'],active=labels.filter(x=>reports[x]),values=active.flatMap(label=>months.map(m=>valueFor(reports[label],m,state,target)?.[metric]).filter(v=>Number.isFinite(v))),range=metricRange(values,metric);const traces=active.map((label,i)=>({type:'scatter',mode:'lines+markers',name:label,x:names,y:months.map(m=>valueFor(reports[label],m,state,target)?.[metric]),line:{color:colors[i%colors.length],width:3},marker:{size:7},hovertemplate:label+'<br>%{x}<br>'+target+' '+metric+': %{y:.6g}<extra></extra>'}));const scope=state==='all'?'all states (pooled)':state;const layout={title:target+' — '+metric+' — '+scope,yaxis:{title:metric,range:range},xaxis:{fixedrange:false},margin:{l:65,r:15,t:45,b:50},paper_bgcolor:'#fff',plot_bgcolor:'#fff',legend:{orientation:'h',y:-.2}};Plotly.react('lineChart',traces,layout,chartConfig)}
function drawContext(state,metric,target,context){const active=labels.filter(x=>reports[x]),all=active.flatMap(label=>(reports[label].site_diagnostics||[]).filter(p=>p.target===target&&(state==='all'||p.context.state===state)).map(p=>p.metrics[metric]).filter(v=>Number.isFinite(v))),range=metricRange(all,metric);const traces=active.map((label,i)=>{const points=(reports[label].site_diagnostics||[]).filter(p=>p.target===target&&(state==='all'||p.context.state===state));const binKey=context+'_bin';return{type:'scatter',mode:'markers',name:label,x:points.map(p=>p.context[context]),y:points.map(p=>p.metrics[metric]),customdata:points.map(p=>[p.site_id,p.context.state,'Q'+(Number(p.context[binKey])+1),p.context[context]]),marker:{color:colors[i%colors.length],size:9,opacity:.72},hovertemplate:label+'<br>site: %{customdata[0]} (%{customdata[1]})<br>'+context+': %{customdata[3]:.6g}<br>split quartile: %{customdata[2]}<br>'+target+' '+metric+': %{y:.6g}<extra></extra>'}});const title=contextSelect.options[contextSelect.selectedIndex].text;const layout={title:target+' '+metric+' by '+title+' — pooled Jun–Sep',xaxis:{title:title,type:context==='cell_count'?'log':'linear'},yaxis:{title:metric,range:range},margin:{l:65,r:15,t:45,b:50},paper_bgcolor:'#fff',plot_bgcolor:'#fff',legend:{orientation:'h',y:-.2}};Plotly.react('contextChart',traces,layout,chartConfig)}
function render(){const month=document.querySelector('#month').value,state=stateSelect.value,metric=document.querySelector('#metric').value,target=targetSelect.value,context=contextSelect.value;let text='<table><thead><tr><th>Model</th>'+targets.map(t=>'<th>'+t+'</th>').join('')+'</tr></thead><tbody>';for(const label of labels){const report=reports[label];text+='<tr'+(!report?' class="pending"':'')+'><td>'+label+(report?'':' — pending')+'</td>';for(const t of targets){const x=valueFor(report,month,state,t);let shown='—';if(x&&x[metric]!==undefined)shown=Number(x[metric]).toFixed(['mae','rmse','bias'].includes(metric)?4:3);text+='<td>'+shown+'</td>'}text+='</tr>'}document.querySelector('#comparison').innerHTML=text+'</tbody></table>';let cards='';for(const label of labels){const r=reports[label];cards+='<div class="card"><strong>'+label+'</strong><br>'+(!r?'<span class="pending">Awaiting validation report</span>':'Report loaded: '+Object.keys(r.by_month||{}).length+' months, '+states.length+' states')+'</div>'}document.querySelector('#cards').innerHTML=cards;drawBars(month,state,metric,target);drawLines(state,metric,target);drawContext(state,metric,target,context)}
document.querySelectorAll('select').forEach(x=>x.addEventListener('change',render));addEventListener('resize',render);render();</script></body></html>"""
    template = template.replace("ECH2O fixed-window recurrent model comparison", "ECH2O recurrent model comparison")
    template = template.replace(
        "Pixel-weighted spatial-validation metrics. Select a month and state to compare identical targets across completed models. Pending models are intentionally blank rather than inferred.",
        "Pixel-weighted spatial-validation metrics. Fixed-window reports use the full 75/25 split; full-water-year BPTT reports use its continuity-valid 290-train / 97-validation subset. Select a month and state to compare targets within a common protocol.",
    )
    page = (template.replace("__PAYLOAD__", payload)
            .replace("__LABELS__", label_json)
            .replace("__PLOTLY_JS__", plotly_js))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    print(args.output)


if __name__ == "__main__":
    main()
