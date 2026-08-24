#!/usr/bin/env python3
"""Build a standalone Plotly explorer for pixel-level terrain/climate strata."""
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
    parser.add_argument("--report", action="append", type=parse_report, required=True,
                        help="Pixel stratum report as LABEL=PATH; repeatable.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {label: json.loads(path.read_text()) for label, path in args.report}
    payload = json.dumps(reports).replace("</", "<\\/")
    plotly_js = get_plotlyjs().replace("</", "<\\/")
    template = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECH2O pixel-strata explorer</title><script>__PLOTLY_JS__</script><style>body{font:15px system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 18px;color:#17212b;background:#fafbfd}.muted{color:#52616d}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}label{font-weight:600}select{margin-left:6px;padding:5px}.plot{background:white;border:1px solid #d7dee5;border-radius:7px;padding:12px}.chart{height:520px}.note{background:#fff7df;padding:10px;border-left:4px solid #d39e00}</style></head><body><h1>ECH2O pixel-level terrain and climate strata</h1><p class="muted">Each bar aggregates every valid held-out pixel within a fixed static-pixel quantile. Static bin edges are shared within a report, and the hover label gives the actual numeric range and valid-pixel count.</p><div class="controls"><label>Model <select id="model"></select></label><label>Month <select id="month"><option value="all">All June–September</option><option value="6">June</option><option value="7">July</option><option value="8">August</option><option value="9">September</option></select></label><label>State <select id="state"><option value="all">All states</option></select></label><label>Context <select id="context"></select></label><label>Target <select id="target"></select></label><label>Metric <select id="metric"><option value="mae">MAE ↓</option><option value="rmse">RMSE ↓</option><option value="bias">Bias</option><option value="correlation">Correlation ↑</option><option value="r2">R² ↑</option><option value="sd_ratio">SD ratio (target 1)</option></select></label></div><div class="note">Interpret these as association diagnostics, not causation: terrain, climate, vegetation, and geography are correlated. Small month/state strata may be noisier; always inspect the valid-pixel count in the hover label.</div><section class="plot"><div id="chart" class="chart"></div></section><script>const reports=__PAYLOAD__;const labels=Object.keys(reports),targets=['soilmoisture','tskin_am','tskin_pm','plc_am','plc_pm'];const model=document.querySelector('#model'),month=document.querySelector('#month'),state=document.querySelector('#state'),context=document.querySelector('#context'),target=document.querySelector('#target'),metric=document.querySelector('#metric');labels.forEach(x=>model.add(new Option(x,x)));targets.forEach(x=>target.add(new Option(x,x)));function options(){const r=reports[model.value];context.innerHTML='';Object.entries(r.contexts).forEach(([key,v])=>context.add(new Option(v.label,key));state.innerHTML='<option value="all">All states</option>';Object.keys(r.by_state||{}).sort().forEach(x=>state.add(new Option(x,x)))}function selectedContexts(r){if(month.value==='all'&&state.value==='all')return r.contexts;if(month.value==='all')return r.by_state?.[state.value]?.contexts;if(state.value==='all')return r.by_month?.[month.value]?.contexts;return r.by_month_state?.[month.value]?.[state.value]?.contexts}function range(values,key){const lo=Math.min(...values),hi=Math.max(...values);if(['mae','rmse'].includes(key))return[0,Math.max(hi*1.15,.001)];if(key==='bias'){const end=Math.max(Math.abs(lo),Math.abs(hi),.001)*1.15;return[-end,end]}const span=Math.max(hi-lo,Math.abs(hi)*.06,.001);return[lo-span*.2,hi+span*.2]}function render(){const r=reports[model.value],contexts=selectedContexts(r),key=metric.value;if(!contexts){Plotly.react('chart',[],{annotations:[{text:'No valid pixels for this month/state selection.',showarrow:false}],paper_bgcolor:'#fff'});return}const c=contexts[context.value],names=Object.keys(c.bins),values=names.map(n=>c.bins[n][target.value][key]),edges=c.edges,custom=names.map((n,i)=>[edges[i],edges[i+1],c.bins[n][target.value].count]);const trace={type:'bar',x:names,y:values,customdata:custom,marker:{color:'#1769aa'},hovertemplate:'%{x}<br>'+c.label+': %{customdata[0]:.6g} to %{customdata[1]:.6g}<br>valid pixels: %{customdata[2]:,}<br>'+target.value+' '+key+': %{y:.6g}<extra></extra>'};const scope=(month.value==='all'?'Jun–Sep':month.options[month.selectedIndex].text)+' / '+(state.value==='all'?'all states':state.value);Plotly.react('chart',[trace],{title:target.value+' '+key+' by '+c.label+' — '+scope,yaxis:{title:key,range:range(values,key)},xaxis:{title:'Static-pixel quantile (Q1 = low, Q4 = high)'},margin:{l:70,r:15,t:55,b:60},paper_bgcolor:'#fff',plot_bgcolor:'#fff'}, {responsive:true,displaylogo:false})}model.addEventListener('change',()=>{options();render()});[month,state,context,target,metric].forEach(x=>x.addEventListener('change',render));options();render();</script></body></html>"""
    # The template remains a compact single string; correct the nested JS call
    # before rendering it so the Context selector populates in a browser.
    template = template.replace(
        "context.add(new Option(v.label,key));state.innerHTML",
        "context.add(new Option(v.label,key)));state.innerHTML",
    )
    # Some state/month combinations have no pixels in a split-wide quartile.
    # Omit those bins rather than showing a misleading zero-error bar.
    template = template.replace(
        "const c=contexts[context.value],names=Object.keys(c.bins),values=",
        "const c=contexts[context.value],names=Object.keys(c.bins).filter(n=>c.bins[n][target.value].count>0),values=",
    )
    template = template.replace(
        "</style></head>",
        ".nav{display:flex;gap:10px;flex-wrap:wrap;padding:0 0 22px;border-bottom:1px solid #d7dee5;margin-bottom:26px}.nav a{color:#17212b;text-decoration:none;padding:7px 11px;border-radius:999px;background:#edf2f7;font-weight:650}.nav a.active,.nav a:hover{background:#1769aa;color:#fff}</style></head>",
    )
    template = template.replace(
        "<body><h1>",
        "<body><nav class=\"nav\"><a href=\"index.html\">Overview</a><a href=\"fixed_window_model_comparison.html\">Model comparison</a><a class=\"active\" href=\"pixel_strata_explorer.html\">Terrain & climate strata</a><a href=\"pixel_joint_heatmap.html\">Deficit × TPI</a></nav><h1>",
    )
    template = template.replace(
        "</p><div class=\"controls\">",
        "</p><p class=\"note\"><strong>Model coverage:</strong> the selector includes only the current calendar-aware ConvLSTM reports. Fixed windows and full-BPTT use the same 114-site spatial-validation cohort; full-BPTT differs by replaying Jan. 1 through Sep. 30 before scoring June–September.</p><div class=\"controls\">",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__PLOTLY_JS__", plotly_js).replace("__PAYLOAD__", payload))
    print(args.output)


if __name__ == "__main__":
    main()
