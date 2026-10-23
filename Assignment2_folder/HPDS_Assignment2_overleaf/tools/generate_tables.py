"""Generate report tables and figures only from independently audited evidence."""
import json, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'data/audited_data.json').read_text())
MAP='maple-baseline-20261022'; A0='athena-baseline-20261022'; A='athena-shared-memory-20261022'; H='athena-h100-single-gpu-20261023'; C='athena-cpu-20261023'
names={MAP:'Maple GTX 1080 Ti, global baseline', A0:'Athena A100, initial global baseline', A:'Athena A100, paired global/shared sweep', H:'Athena H100, paired global/shared sweep', C:'Athena EPYC 7742, CPU comparison'}

def summary(tag,d,s,b):
 key='program' if tag==C else 'strategy'; p='workers' if tag==C else 'block_size'
 return next(r for r in data[tag]['summary'] if r['dataset']==d and r[key]==s and int(r[p])==b)

def mean(tag,d,s,b):return float(summary(tag,d,s,b)['mean_ms'])

def tab(caption,cols,header,rows,label):
 return '\\begin{table}[t]\n\\centering\\caption{'+caption+'}\\label{'+label+'}\n\\small\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{'+cols+'}\\toprule\n'+header+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(r)+r' \\' for r in rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}\n'

best=[]
for tag,short in [(MAP,'GTX 1080 Ti'),(A,'A100'),(H,'H100')]:
 for d in ['small','medium','large']:
  r=min((r for r in data[tag]['summary'] if r['dataset']==d and r['strategy']!='serial'),key=lambda r:float(r['mean_ms']))
  best.append([short,d,r['strategy'],r['block_size'],f"{float(r['mean_ms']):.3f}",f"{float(r['serial_speedup']):.2f}"])
(ROOT/'data/best.tex').write_text(tab('Lowest observed mean GPU classification time per dataset. Speedup uses serial from the same sweep.','lllrrr','GPU & Data & Kernel & Block & ms & Speedup',best,'tab:best'))
rows=[]
for d in ['small','medium','large']:
 for strategy in ['serial','pthreads','openmp','mpi']:
  r=min((r for r in data[C]['summary'] if r['dataset']==d and r['program']==strategy),key=lambda r:float(r['mean_ms']))
  rows.append([d,strategy,r['workers'],f"{float(r['mean_ms']):.3f}",f"{float(r['min_ms']):.3f}",f"{float(r['max_ms']):.3f}"])
(ROOT/'data/cpu_best.tex').write_text(tab('CPU configurations with lowest observed mean, EPYC 7742. Range spans three runs.','llrrrr','Data & Program & Workers & Mean & Min & Max',rows,'tab:cpu'))
rows=[]
for d in ['small','medium','large']:
 gpu=mean(H,d,'shared',128)
 rows.append([d,f'{gpu:.3f}']+[f'{mean(C,d,s,1 if s=="serial" else 16)/gpu:.2f}' for s in ['serial','pthreads','openmp','mpi']])
(ROOT/'data/cross.tex').write_text(tab('Cross-platform CPU/GPU time ratios. CPU uses EPYC 7742 (16 workers for parallel programs); GPU uses H100 with Xeon 8462Y+ host. Ratios are not same-machine speedups.','lrrrrr','Data & GPU ms & Serial & Pthreads & OpenMP & MPI',rows,'tab:cross'))

# Individual runs and averages, including every completed configuration.
append=[]
for tag in [MAP,A0,A,H,C]:
 cpu=tag==C; key='program' if cpu else 'strategy'; p='workers' if cpu else 'block_size'
 groups=defaultdict(list)
 for r in data[tag]['raw']:groups[(r['dataset'],r[key],r[p])].append(r)
 append.append('\\subsection{'+names[tag]+'}\n')
 append.append('\\small\n\\begin{longtable}{lllrrrrrr}\n\\caption{'+names[tag]+'. Individual classification times and means in ms. Three repetitions per row.}\\\\\n\\toprule\nData & '+('Program & Workers' if cpu else 'Kernel & Block')+' & Run 1 & Run 2 & Run 3 & Mean & '+('Speedup' if cpu else 'Kernel mean')+' & '+('Efficiency' if cpu else 'Speedup')+' \\\\\n\\midrule\\endfirsthead\n\\toprule\nData & '+('Program & Workers' if cpu else 'Kernel & Block')+' & Run 1 & Run 2 & Run 3 & Mean & '+('Speedup' if cpu else 'Kernel mean')+' & '+('Efficiency' if cpu else 'Speedup')+' \\\\\n\\midrule\\endhead\n\\bottomrule\\endfoot\n')
 for r in data[tag]['summary']:
  g=sorted(groups[r['dataset'],r[key],r[p]],key=lambda v:int(v['rep']))
  vals=[r['dataset'],r[key],r[p]]+[f"{float(v['end_to_end_ms']):.6f}" for v in g]+[f"{float(r['mean_ms']):.6f}"]
  if cpu:vals += [f"{float(r['speedup_vs_serial']):.4f}",f"{float(r['efficiency_vs_1_worker']):.4f}" if r.get('efficiency_vs_1_worker') else '--']
  else:vals += [f"{statistics.mean(float(v['kernel_ms']) for v in g):.6f}" if r[key]!='serial' else '--',f"{float(r['serial_speedup']):.4f}"]
  append.append(' & '.join(vals)+r' \\'+'\n')
 append.append('\\end{longtable}\\normalsize\n')
(ROOT/'data/all_runs.tex').write_text(''.join(append))

# Large-dataset strategy comparisons keep each platform explicitly labeled.
plot=r'''\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=5.4cm,xlabel={Threads per block},ylabel={Classification time (ms)},xtick={64,128,256},legend style={font=\scriptsize,at={(0.5,1.02)},anchor=south,legend columns=2},grid=major]
'''
for tag,platform,color in [(A,'A100','blue'),(H,'H100','red')]:
 for s,style in [('global','dashed'),('shared','solid')]:
  coords=' '.join(f'({b},{mean(tag,"large",s,b)})' for b in [64,128,256])
  plot+=f'\\addplot[{color},{style},mark={"square*" if s=="shared" else "o"}] coordinates {{{coords}}};\n\\addlegendentry{{{platform} {s}}}\n'
plot+=r'''\end{axis}\end{tikzpicture}
\caption{Large-dataset block-size experiment, three-run means from the paired sweeps. Hardware differs between A100 and H100.}\label{fig:gpu}
\end{figure}
'''
(ROOT/'data/gpu_plot.tex').write_text(plot)
plot=r'''\begin{figure}[t]\centering
\begin{tikzpicture}\begin{axis}[width=0.97\columnwidth,height=5.1cm,xlabel={CPU workers},ylabel={Time (ms)},xmode=log,log basis x=2,xtick={1,2,4,8,16},xticklabels={1,2,4,8,16},legend style={font=\scriptsize},grid=major]
'''
for s,color in [('pthreads','blue'),('openmp','red'),('mpi','black')]:
 coords=' '.join(f'({w},{mean(C,"large",s,w)})' for w in [1,2,4,8,16])
 plot+=f'\\addplot[{color},mark=*] coordinates {{{coords}}};\\addlegendentry{{{s}}}\n'
plot+=r'''\end{axis}\end{tikzpicture}
\caption{Large-dataset CPU classification times on Athena's EPYC 7742 node. Three-run means are shown; individual values in the appendix expose run-to-run variation.}\label{fig:cpu}
\end{figure}
'''
(ROOT/'data/cpu_plot.tex').write_text(plot)
print('Generated tables and plots for all 342 measured runs')
