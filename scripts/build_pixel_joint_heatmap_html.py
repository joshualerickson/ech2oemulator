#!/usr/bin/env python3
"""Build an interactive Plotly heatmap for joint pixel terrain/climate strata."""
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
    parser.add_argument("--report", type=parse_report, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {label: json.loads(path.read_text()) for label, path in args.report}
    payload = json.dumps(reports).replace("</", "<\\/")
    plotly_js = get_plotlyjs().replace("</", "<\\/")
    template = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECH2O joint pixel-strata heatmap</title><script>__PLOTLY_JS__</script><style>body{font:15px system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 18px;color:#17212b;background:#fafbfd}.muted{color:#52616d}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}label{font-weight:600}select{margin-left:6px;padding:5px}.plot{background:#fff;border:1px solid #d7dee5;border-radius:7px;padding:12px}.chart{height:650px}details{background:#eef4f8;padding:12px;margin:14px 0}dt{font-weight:700}dd{margin:0 0 10px}</style><body><h1>ECH2O joint pixel-strata heatmap</h1><p class="muted">The heatmap diagnoses joint terrain–climate regimes. Each cell aggregates all valid held-out pixels belonging to both split-wide static quartiles.</p><div class="controls"><label>Model <select id="model"></select></label><label>Month <select id="month"><option value="all">All June–September</option><option value="6">June</option><option value="7">July</option><option value="8">August</option><option value="9">September</option></select></label><label>State <select id="state"><option value="all">All states</option></select></label><label>Target <select id="target"></select></label><label>Metric <select id="metric"><option value="mae">MAE ↓</option><option value="rmse">RMSE ↓</option><option value="bias">Bias</option><option value="correlation">Correlation ↑</option><option value="r2">R² ↑</option><option value="sd_ratio">SD ratio (target 1)</option></select></label></div><details open><summary><strong>How the metrics are calculated</strong></summary><dl><dt>MAE</dt><dd>Mean absolute prediction error across every valid pixel in the selected heatmap cell.</dd><dt>RMSE</dt><dd>Square root of mean squared pixel error; larger errors receive more weight.</dd><dt>Bias</dt><dd>Mean signed residual, prediction minus observation.</dd><dt>Correlation / R² / SD ratio</dt><dd>Pooled pixel-pair statistics, not averages of site metrics. SD ratio is prediction standard deviation divided by observation standard deviation.</dd><dt>Counts and interpretation</dt><dd>Hover for the valid-pixel count and physical ranges. Empty cells are shown as blank. This identifies associations, not causal effects, because terrain, climate, vegetation, and geography covary.</dd></dl></details><section class="plot"><div id="chart" class="chart"></div></section><script>const reports=__PAYLOAD__,labels=Object.keys(reports),targets=['soilmoisture','tskin_am','tskin_pm','plc_am','plc_pm'];const model=document.querySelector('#model'),month=document.querySelector('#month'),state=document.querySelector('#state'),target=document.querySelector('#target'),metric=document.querySelector('#metric');labels.forEach(x=>model.add(new Option(x,x)));targets.forEach(x=>target.add(new Option(x,x)));function options(){const r=reports[model.value];state.innerHTML='<option value="all">All states</option>';Object.keys(r.joint_by_state||{}).sort().forEach(x=>state.add(new Option(x,x)))}function jointFor(r){if(month.value==='all'&&state.value==='all')return r.joint_contexts;if(month.value==='all')return r.joint_by_state?.[state.value]?.joint_contexts;if(state.value==='all')return r.joint_by_month?.[month.value]?.joint_contexts;return r.joint_by_month_state?.[month.value]?.[state.value]?.joint_contexts}function render(){const r=reports[model.value],all=jointFor(r),j=all?.wbdef_x_tpi;if(!j){Plotly.react('chart',[],{annotations:[{text:'No valid pixels for this selection.',showarrow:false}]});return}const n=j.x_edges.length-1,z=[],custom=[];for(let yi=0;yi<n;yi++){z[yi]=[];custom[yi]=[];for(let xi=0;xi<n;xi++){const q=j.cells['Q'+(xi+1)+'_Q'+(yi+1)][target.value];z[yi][xi]=q.count?q[metric]:null;custom[yi][xi]=[j.x_edges[xi],j.x_edges[xi+1],j.y_edges[yi],j.y_edges[yi+1],q.count]}}const scope=(month.value==='all'?'Jun–Sep':month.options[month.selectedIndex].text)+' / '+(state.value==='all'?'all states':state.value);Plotly.react('chart',[{type:'heatmap',x:['Q1','Q2','Q3','Q4'],y:['Q1','Q2','Q3','Q4'],z:z,customdata:custom,colorscale:'Viridis',hoverongaps:false,hovertemplate:j.x_label+': %{customdata[0]:.6g} to %{customdata[1]:.6g}<br>'+j.y_label+': %{customdata[2]:.6g} to %{customdata[3]:.6g}<br>valid pixels: %{customdata[4]:,}<br>'+target.value+' '+metric.value+': %{z:.6g}<extra></extra>',colorbar:{title:metric.value}}],{title:target.value+' '+metric.value+' — '+j.x_label+' × '+j.y_label+' — '+scope,xaxis:{title:j.x_label+' quantile (low → high)'},yaxis:{title:j.y_label+' quantile (low → high)'},margin:{l:80,r:90,t:60,b:75},paper_bgcolor:'#fff',plot_bgcolor:'#fff'}, {responsive:true,displaylogo:false})}model.addEventListener('change',()=>{options();render()});[month,state,target,metric].forEach(x=>x.addEventListener('change',render));options();render();</script></body></html>"""
    template = template.replace("q.count?q[metric]:null", "q.count?q[metric.value]:null")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__PLOTLY_JS__", plotly_js).replace("__PAYLOAD__", payload))
    print(args.output)


if __name__ == "__main__":
    main()
