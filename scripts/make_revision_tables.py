from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parents[1];D=P/'evidence';T=P/'tables';T.mkdir(exist_ok=True)
m=pd.read_csv(D/'e17/manuscript_S1_complete_main_table.csv');pairs=pd.read_csv(D/'e17/manuscript_primary_pairs_both_scores.csv')
tr=pd.read_csv(D/'protocols/S1_training_support.csv');bp=m[m.Method.eq('Raw')].merge(tr,on='Origin');bp['TrainingConstant']=bp.TrainPositives/bp.TrainN;bp['ConstantBrier']=bp.ActualRate*(1-bp.TrainingConstant)**2+(1-bp.ActualRate)*bp.TrainingConstant**2
# Evidence files are immutable; the computed constant is used only in tables below.
fmt=lambda x:f'{x:.6f}'
sign=lambda x:f'{x:+.6f}'
ci=lambda r,score='Brier':f"{sign(r[score+'_Delta'])} [{sign(r[score+'_Low'])}, {sign(r[score+'_High'])}]"
text=r'''\begin{table*}[!t]
\caption{Brier scores in the S1 four-origin completed-delivery cohort. TrainConst uses the corresponding training prevalence; PoolRate uses the same daily admitted history as Expanding. Lower scores are better. Values are rounded after full-precision calculation; no cross-origin pooling is used.}
\label{tab:main}\centering\footnotesize\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrrrrrrrr}\toprule
Origin & TrainConst & Raw & Frozen & Expanding & PoolRate & Platt & BrierOpt & PeriodicLR\\\midrule
'''
for o in ['O1','O2','O3','O4']:
 row=m[m.Origin.eq(o)].set_index('Method');text+=o+' & '+fmt(bp.set_index('Origin').loc[o,'ConstantBrier'])+' & '+' & '.join(fmt(row.loc[k,'Brier']) for k in ['Raw','Frozen','Expanding','PoolRate','Platt','BrierOpt','PeriodicLR'])+r' \\'+'\n'
text+=r'''\bottomrule\end{tabular}
\begin{flushleft}\footnotesize PeriodicLR refits the base model at evaluation start and day 30; unlike the other columns, it changes the backbone updating policy. Complete log loss, AP, probability means and paired intervals are in Supplement S4.\end{flushleft}
\end{table*}
''';(T/'main_results.tex').write_text(text)
text=r'''\begin{table}[t]
\caption{S1 historical-window Brier contrasts. Each cell gives the paired difference and conditional 95\% interval. First: Rolling-60 minus Expanding. Second: Approval-60 minus Rolling-60.}
\label{tab:windows}\centering\scriptsize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lcc}\toprule Origin & First contrast & Second contrast\\\midrule
'''
for o in ['O1','O2','O3','O4']:
 out=[]
 for a,b in [('Rolling60','Expanding'),('Approval60','Rolling60')]:
  r=pairs[(pairs.Origin==o)&(pairs.A==a)&(pairs.B==b)].iloc[0]
  out.append(r'\shortstack{'+sign(r.Brier_Delta)+r'\\{}['+sign(r.Brier_Low)+', '+sign(r.Brier_High)+']}')
 text+=o+' & '+' & '.join(out)+r' \\'+'\n'
text+=r'\bottomrule\end{tabular}\end{table}'+'\n';(T/'window_results.tex').write_text(text)
text=r'''\begin{table}[t]
\caption{Outcome-specific Brier changes for S1 Expanding minus Raw. Both contributions use the full origin sample as denominator. This is an algebraic loss partition, not a reliability--resolution decomposition.}
\label{tab:decomp}\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrr}\toprule Origin & On time & Delayed & Total\\\midrule
'''
for o in ['O1','O2','O3','O4']:
 x=m[m.Origin.eq(o)].set_index('Method');v=x.loc['Expanding',['OnTimeBrierContribution','LateBrierContribution','Brier']]-x.loc['Raw',['OnTimeBrierContribution','LateBrierContribution','Brier']];text+=o+' & '+' & '.join(sign(t) for t in v)+r' \\'+'\n'
text+=r'\bottomrule\end{tabular}\end{table}'+'\n';(T/'loss_decomposition.tex').write_text(text)
s=pd.read_csv(D/'e17/decision_metrics.csv');p=pd.read_csv(D/'e17/paired_intervals.csv')
text=r'''\begin{table*}[t]
\caption{Same-protocol S1 screening at a nominal 5\% daily budget. Coverage is the percentage of delayed-order transaction value selected. Differences and intervals are in percentage points; all conditional intervals for Expanding minus Raw include or touch zero.}
\label{tab:screen}\centering\footnotesize\setlength{\tabcolsep}{7pt}
\begin{tabular}{lrrrrl}\toprule Origin & Selected & Raw & Expanding & Value-only & Expanding minus Raw [95\% interval]\\\midrule
'''
for o in ['O1','O2','O3','O4']:
 x=s[(s.Backbone=='Core')&(s.Origin==o)].set_index('Method');r=p[(p.Backbone=='Core')&(p.Origin==o)&(p.BlockDays==7)&(p.A=='Expanding')&(p.B=='Raw')&(p.Metric=='ValueCoverage5')].iloc[0]
 text+=o+f" & {int(x.loc['Raw','Selected'])} & "+' & '.join(f'{100*x.loc[k,"ValueCoverage5"]:.3f}' for k in ['Raw','Expanding','ValueOnly'])+f' & {100*r.Delta:+.3f} [{100*r.CI_low:+.3f}, {100*r.CI_high:+.3f}]'+r' \\'+'\n'
text+=r'\bottomrule\end{tabular}\end{table*}'+'\n';(T/'screening.tex').write_text(text)
print('tables built')

# Complete the initial/update/total score-difference table from the same frozen evidence.
text=r"""\begin{table*}[t]
\caption{Initial correction, continued updating and total correction under Core LR/S1. Entries are Brier differences and conditional 95\% paired seven-day block intervals. Their point estimates add algebraically; interval endpoints do not. Complete log-loss counterparts are in Supplement S4.}
\label{tab:triplets}\centering\footnotesize\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrrr}\toprule
Origin & Frozen minus Raw & Expanding minus Frozen & Expanding minus Raw\\\midrule
"""
for o in ['O1','O2','O3','O4']:
 cells=[]
 for aa,bb in [('Frozen','Raw'),('Expanding','Frozen'),('Expanding','Raw')]:
  rr=pairs[(pairs.Origin==o)&(pairs.A==aa)&(pairs.B==bb)].iloc[0]
  cells.append(r'\shortstack{'+sign(rr.Brier_Delta)+r'\\{}['+sign(rr.Brier_Low)+', '+sign(rr.Brier_High)+']}')
 text+=o+' & '+' & '.join(cells)+r'\\'+'\n'
text+=r'\bottomrule\end{tabular}\end{table*}'+'\n'
(T/'core_triplets.tex').write_text(text)
