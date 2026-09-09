#!/usr/bin/env python3
"""Build the public, observed-manifold VCF--LAI sensitivity dashboard."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PUBLIC_SCENARIOS = ("coupled_low", "coupled_high", "vcf_high_lai_low", "vcf_low_lai_high")


def summarise(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["scenario"] in PUBLIC_SCENARIOS:
            value = float(row["mean_prediction_change"])
            # A global quartile need not occur within every small sampled bbox.
            # Exclude its zero-support (NaN) site/date rows rather than allowing
            # one absent stratum to erase the aggregate bar.
            if np.isfinite(value):
                grouped[(row["scenario"], row["target"], str(int(row["month"])), row["stratum"])].append(value)
    return [
        {"scenario": s, "target": t, "month": m, "stratum": st, "n": len(v),
         "mean": float(np.mean(v)), "median": float(np.median(v)),
         "p10": float(np.quantile(v, .1)), "p90": float(np.quantile(v, .9))}
        for (s, t, m, st), v in sorted(grouped.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--strata-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.dumps({"responses": summarise(rows), "strata": json.loads(args.strata_json.read_text())["quantiles"]}, separators=(",", ":"))
    html = '''<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECH2O canopy sensitivity</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>
body{font:15px system-ui;max-width:1500px;margin:28px auto;padding:0 18px;color:#17212b;background:#f7f9fc}.nav{display:flex;gap:9px;flex-wrap:wrap;padding-bottom:18px;border-bottom:1px solid #d7dee5}.nav a{color:#17212b;text-decoration:none;background:#eaf0f6;padding:7px 11px;border-radius:999px;font-weight:650}.nav a:hover,.nav .active{background:#1769aa;color:#fff}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}label{font-weight:650}select{margin-left:5px;padding:5px}.card{background:#fff;border:1px solid #d7dee5;border-radius:10px;padding:12px;margin:18px 0}.chart{height:560px}.note{background:#fff7df;border-left:4px solid #d39e00;padding:12px;margin:16px 0}
</style><nav class="nav"><a href="index.html">Overview</a><a href="fixed_window_model_comparison.html">Model comparison</a><a href="pixel_strata_explorer.html">Terrain &amp; climate strata</a><a href="pixel_joint_heatmap.html">Deficit × TPI</a><a class="active" href="static_feature_sensitivity.html">Canopy sensitivity</a></nav>
<h1>Coupled canopy sensitivity</h1><p>Forcings and all non-canopy statics are fixed. Each bar is the prediction change from the same pixel’s baseline—not a causal intervention estimate.</p>
<div class="note"><b>Canopy configurations.</b> Coupled low is VCF −1 SD and LAI −1 SD; coupled high is VCF +1 SD and LAI +1 SD. The two mixed configurations change one feature upward and the other downward. VCF and LAI are strongly correlated in training data, so read mixed configurations as model diagnostics. Bars are grouped by month, never stacked. Quartiles are global training-pixel ranges; hover to see physical bounds.</div>
<div class="controls"><label>Target <select id="target"></select></label><label>Canopy configuration <select id="scenario"><option value="coupled_high">Coupled high</option><option value="coupled_low">Coupled low</option><option value="vcf_high_lai_low">VCF high / LAI low</option><option value="vcf_low_lai_high">VCF low / LAI high</option></select></label><label>Stratum family <select id="family"></select></label><label>Statistic <select id="stat"><option value="mean">Mean response</option><option value="median">Median response</option></select></label></div>
<section class="card"><h2>Canopy-configuration response by month</h2><div id="response" class="chart"></div></section>
<section class="card"><h2>How to read this diagnostic</h2><ul><li><b>Positive</b> values mean the emulator predicts a higher target than its unmodified baseline; negative values mean lower.</li><li>These are <b>pixel-level</b> perturbations, averaged over valid pixels within each sampled site/date and then summarized across samples.</li><li>The P10–P90 hover interval describes variation across sampled site/date responses. An interval spanning zero means the response is not directionally consistent across those contexts.</li><li>For global DEM, TPI, TWI, and deficit quartiles, the hover sample count is the number of sampled site/dates that actually contain pixels in that quartile. Small bboxes may not contain every global quartile.</li><li>VCF and LAI are observationally correlated, so this supports interpretation of their shared canopy-gradient response—not attribution to one vegetation variable.</li></ul></section>
<script>const D=__DATA__,T=['soilmoisture','tskin_am','tskin_pm','plc_am','plc_pm'],MN={6:'June',7:'July',8:'August',9:'September'},target=document.querySelector('#target'),scenario=document.querySelector('#scenario'),family=document.querySelector('#family'),stat=document.querySelector('#stat');T.forEach(x=>target.add(new Option(x,x)));['all','dem','tpi','twi','wbdef'].forEach(x=>family.add(new Option(x==='all'?'All valid pixels':x.toUpperCase()+' quartiles',x)));function labs(){return family.value==='all'?['all']:['q1','q2','q3','q4']}function bound(q){if(family.value==='all')return 'All valid pixels';let z=D.strata[family.value],e=[-Infinity,z.q25,z.q50,z.q75,Infinity],i=+q.slice(1)-1,f=x=>Number.isFinite(x)?x.toPrecision(5):(x<0?'−∞':'∞');return family.value.toUpperCase()+' Q'+(i+1)+': '+f(e[i])+' to '+f(e[i+1])}function render(){let rows=D.responses.filter(r=>r.target===target.value&&r.scenario===scenario.value&&(family.value==='all'?r.stratum==='all':r.stratum.startsWith(family.value+'_'))),xs=labs(),ms=[...new Set(rows.map(r=>r.month))].sort((a,b)=>a-b),tr=ms.map(m=>{let mp=Object.fromEntries(rows.filter(r=>r.month===m).map(r=>[r.stratum,r])),yy=xs.map(q=>mp[q==='all'?'all':family.value+'_'+q]?.[stat.value]??null),cd=xs.map(q=>{let r=mp[q==='all'?'all':family.value+'_'+q];return r?[r.n,r.p10,r.p90,bound(q)]:null});return{type:'bar',name:MN[m],x:xs.map(x=>x.toUpperCase()),y:yy,customdata:cd,hovertemplate:'%{x}<br>'+MN[m]+'<br>'+stat.value+': %{y:.4g}<br>site/date samples: %{customdata[0]}<br>P10–P90: %{customdata[1]:.4g} to %{customdata[2]:.4g}<br>%{customdata[3]}<extra></extra>'}}),v=tr.flatMap(x=>x.y).filter(x=>x!==null),r=Math.max(...v.map(Math.abs),.001)*1.15;Plotly.react('response',tr,{barmode:'group',yaxis:{title:'Prediction change from baseline',range:[-r,r]},xaxis:{title:family.value==='all'?'Support':family.value.toUpperCase()+' global training-pixel quartile',ticktext:xs.map(bound),tickvals:xs.map(x=>x.toUpperCase())},shapes:[{type:'line',x0:-.5,x1:xs.length-.5,y0:0,y1:0,line:{color:'#555'}}],legend:{orientation:'h'},margin:{l:70,r:15,t:35,b:110},paper_bgcolor:'#fff',plot_bgcolor:'#fff'},{responsive:true,displaylogo:false})}[target,scenario,family,stat].forEach(x=>x.addEventListener('change',render));render();</script>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html.replace("__DATA__", payload), encoding="utf-8")
    print({"rows": len(rows), "published_scenarios": list(PUBLIC_SCENARIOS), "output": str(args.output)})


if __name__ == "__main__":
    main()
