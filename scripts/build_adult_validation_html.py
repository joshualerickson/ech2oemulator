#!/usr/bin/env python3
"""Build a standalone dashboard for the changed adult-response validation run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plotly.offline import get_plotlyjs


METRICS = ("mae", "rmse", "bias", "correlation", "r2", "sd_ratio")
TARGETS = ("soilmoisture", "tskin_am", "tskin_pm", "plc_am", "plc_pm")


def read_metrics(path: Path) -> list[dict[str, str | int | float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["count"] = int(row["count"])
        for name in METRICS:
            row[name] = float(row[name])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.metrics_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["split"] != "dev_spatial_val" or Path(summary["manifest"]) != args.manifest:
        raise ValueError("Metrics must come from the requested adult spatial-validation manifest")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        validation_rows = [row for row in csv.DictReader(handle) if row["split"] == "dev_spatial_val"]
    if not validation_rows or any(not row["target_source"].endswith("_subdaily_adult.nc") for row in validation_rows):
        raise ValueError("The selected validation rows must point to adult response files")
    tables = {
        "overall": read_metrics(args.metrics_dir / "by_target.csv"),
        "month": read_metrics(args.metrics_dir / "by_month_target.csv"),
        "state": read_metrics(args.metrics_dir / "by_state_target.csv"),
        "month_state": read_metrics(args.metrics_dir / "by_month_state_target.csv"),
        "site": read_metrics(args.metrics_dir / "by_site_target.csv"),
    }
    if {row["target"] for row in tables["overall"]} != set(TARGETS):
        raise ValueError("Adult report does not contain all five expected targets")
    payload = json.dumps({
        "tables": tables,
        "epoch": summary["checkpoint_epoch"],
        "site_count": len({row["site_id"] for row in validation_rows}),
        "window_count": len(validation_rows),
        "months": summary["months"],
    }, separators=(",", ":")).replace("</", "<\\/")
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECH2O adult-response validation</title><script>__PLOTLY__</script><style>
body{font:15px system-ui,sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;color:#17212b;background:#f7f9fc;line-height:1.45}h1{margin-bottom:4px}.muted{color:#52616d}.nav{display:flex;gap:10px;flex-wrap:wrap;padding:0 0 20px;border-bottom:1px solid #d7dee5;margin-bottom:24px}.nav a{color:#17212b;text-decoration:none;padding:7px 11px;border-radius:999px;background:#edf2f7;font-weight:650}.nav a.active,.nav a:hover{background:#1769aa;color:#fff}.note{background:#fff7df;border-left:4px solid #d39e00;padding:12px 14px;margin:18px 0}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}label{font-weight:650}select{margin-left:5px;padding:5px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.card{background:#fff;border:1px solid #d7dee5;border-radius:9px;padding:12px;margin:12px 0}.chart{height:420px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #d7dee5;padding:7px 9px;text-align:right}th{background:#edf2f7}th:first-child,td:first-child{text-align:left}@media(max-width:600px){.grid{display:block}.chart{height:360px}}
</style></head><body><nav class="nav"><a href="index.html">Overview</a><a href="fixed_window_model_comparison.html">Original model comparison</a><a class="active" href="adult_validation.html">Adult validation</a><a href="pixel_strata_explorer.html">Terrain &amp; climate strata</a><a href="pixel_joint_heatmap.html">Deficit × TPI</a><a href="static_feature_sensitivity.html">Canopy sensitivity</a></nav>
<h1>Adult-response ConvLSTM validation</h1><p class="muted" id="protocol"></p>
<div class="note"><strong>Separate response dataset.</strong> These metrics compare the adult ET-stress model with adult ECH2O targets on held-out sites. The original-model comparison uses earlier target values; its scores should not be ranked against this page as if the truth were identical. The extra NetCDF variable <code>et</code> is not a model target. Adult target-artifact triage should be reviewed before treating these scores as final scientific evidence.</div>
<div class="controls"><label>Target <select id="target"></select></label><label>Metric <select id="metric"><option value="mae">MAE ↓</option><option value="rmse">RMSE ↓</option><option value="bias">Bias</option><option value="correlation">Correlation ↑</option><option value="r2">R² ↑</option><option value="sd_ratio">SD ratio (target 1)</option></select></label><label>Month <select id="month"><option value="all">All June–September</option><option value="6">June</option><option value="7">July</option><option value="8">August</option><option value="9">September</option></select></label><label>State <select id="state"><option value="all">All states</option></select></label></div>
<div class="grid"><section class="card"><h2>Seasonal trajectory</h2><div id="seasonal" class="chart"></div></section><section class="card"><h2>Across states</h2><div id="states" class="chart"></div></section></div>
<section class="card"><h2>Selected month and state</h2><p class="muted">Exact pooled pixel metrics for all five targets. Counts are valid prediction/target pixel pairs.</p><div id="targetTable"></div></section>
<section class="card"><h2>Site spread</h2><p class="muted">Each point is one held-out site, pooled across June–September. Use the hover labels to identify outlying sites; the table above remains pixel pooled.</p><div id="sites" class="chart"></div></section>
<section class="card"><h2>Reading the metrics</h2><p>MAE and RMSE are absolute pixel errors in each target's native units. Bias is prediction minus adult target. Correlation, R², and the predicted/observed standard-deviation ratio are pooled over valid pixels in the selected group. State and month summaries are recomputed from their own pixel pairs; they are not averages of site scores.</p><p class="muted">The selected checkpoint is epoch 33. Its training validation loss selected the checkpoint; the metrics on this page are calculated separately from its saved predictions against adult targets.</p></section>
<script>const D=__DATA__,N={6:'June',7:'July',8:'August',9:'September'},T=['soilmoisture','tskin_am','tskin_pm','plc_am','plc_pm'],target=document.querySelector('#target'),metric=document.querySelector('#metric'),month=document.querySelector('#month'),state=document.querySelector('#state');T.forEach(x=>target.add(new Option(x,x)));[...new Set(D.tables.state.map(x=>x.state))].sort().forEach(x=>state.add(new Option(x,x)));document.querySelector('#protocol').textContent=`Best checkpoint: epoch ${D.epoch} · 90-day forcing history · ${D.site_count} unseen sites · ${D.window_count.toLocaleString()} June–September site/day windows`;
function one(rows,t,m,s){return rows.find(x=>x.target===t&&(m===undefined||x.month===m)&&(s===undefined||x.state===s))||null}function selected(t){const m=month.value,s=state.value;if(m==='all'&&s==='all')return one(D.tables.overall,t);if(m==='all')return one(D.tables.state,t,undefined,s);if(s==='all')return one(D.tables.month,t,m);return one(D.tables.month_state,t,m,s)}function range(values,k){let v=values.filter(Number.isFinite);if(!v.length)return[0,1];let lo=Math.min(...v),hi=Math.max(...v);if(k==='mae'||k==='rmse')return[0,Math.max(hi*1.12,.001)];if(k==='bias'){let a=Math.max(Math.abs(lo),Math.abs(hi),.001)*1.12;return[-a,a]}let gap=Math.max(hi-lo,Math.abs(hi)*.07,.001);return[lo-gap*.2,hi+gap*.2]}function fmt(v){return v===null||v===undefined?'—':Number(v).toFixed(4)}function render(){let k=metric.value,t=target.value,s=state.value,m=month.value,ms=['6','7','8','9'],season=ms.map(x=>s==='all'?one(D.tables.month,t,x):one(D.tables.month_state,t,x,s)),states=[...new Set(D.tables.state.map(x=>x.state))].sort(),stateRows=states.map(x=>m==='all'?one(D.tables.state,t,undefined,x):one(D.tables.month_state,t,m,x)),siteRows=D.tables.site.filter(x=>x.target===t&&(s==='all'||x.site_id.startsWith(s))).sort((a,b)=>a.site_id.localeCompare(b.site_id));Plotly.react('seasonal',[{type:'scatter',mode:'lines+markers',x:ms.map(x=>N[x]),y:season.map(x=>x?.[k]??null),customdata:season.map(x=>x?.count??null),line:{color:'#1769aa',width:3},hovertemplate:'%{x}<br>'+t+' '+k+': %{y:.5g}<br>valid pixels: %{customdata:,}<extra></extra>'}],{yaxis:{title:k,range:range(season.map(x=>x?.[k]),k)},margin:{l:75,r:20,t:20,b:55},paper_bgcolor:'#fff',plot_bgcolor:'#fff'},{responsive:true,displaylogo:false});Plotly.react('states',[{type:'bar',x:states,y:stateRows.map(x=>x?.[k]??null),customdata:stateRows.map(x=>x?.count??null),marker:{color:'#1b9e77'},hovertemplate:'%{x}<br>'+t+' '+k+': %{y:.5g}<br>valid pixels: %{customdata:,}<extra></extra>'}],{yaxis:{title:k,range:range(stateRows.map(x=>x?.[k]),k)},margin:{l:75,r:20,t:20,b:65},paper_bgcolor:'#fff',plot_bgcolor:'#fff'},{responsive:true,displaylogo:false});Plotly.react('sites',[{type:'scatter',mode:'markers',x:siteRows.map(x=>x.site_id),y:siteRows.map(x=>x[k]),customdata:siteRows.map(x=>x.count),marker:{color:'#7570b3',size:8,opacity:.7},hovertemplate:'site: %{x}<br>'+t+' '+k+': %{y:.5g}<br>valid pixels: %{customdata:,}<extra></extra>'}],{yaxis:{title:k,range:range(siteRows.map(x=>x[k]),k)},xaxis:{title:'Held-out sites',showticklabels:false},margin:{l:75,r:20,t:20,b:55},paper_bgcolor:'#fff',plot_bgcolor:'#fff'},{responsive:true,displaylogo:false});let html='<table><thead><tr><th>Target</th><th>Valid pixels</th><th>MAE</th><th>RMSE</th><th>Bias</th><th>Correlation</th><th>R²</th><th>SD ratio</th></tr></thead><tbody>';for(let name of T){let x=selected(name);html+='<tr><td>'+name+'</td><td>'+(x?x.count.toLocaleString():'—')+'</td>'+['mae','rmse','bias','correlation','r2','sd_ratio'].map(q=>'<td>'+(x?fmt(x[q]):'—')+'</td>').join('')+'</tr>'}document.querySelector('#targetTable').innerHTML=html+'</tbody></table>'}[target,metric,month,state].forEach(x=>x.addEventListener('change',render));render();</script></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html.replace("__PLOTLY__", get_plotlyjs().replace("</", "<\\/")).replace("__DATA__", payload), encoding="utf-8")
    print({"output": str(args.output), "epoch": summary["checkpoint_epoch"], "sites": len({row["site_id"] for row in validation_rows}), "windows": len(validation_rows)})


if __name__ == "__main__":
    main()
